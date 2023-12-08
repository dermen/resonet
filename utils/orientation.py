
import os
from copy import deepcopy
import numpy as np
import torch
from scipy.spatial.transform import Rotation

try:
    HAS_CCTBX=True
    from cctbx import sgtbx, uctbx
    from cctbx.sgtbx.literal_description import  literal_description
    from scitbx.matrix import sqr
    from iotbx import pdb as iotbx_pdb
    from simtbx.nanoBragg import nanoBragg
    from dxtbx.model import Crystal, DetectorFactory, BeamFactory
    from resonet.sims import process_pdb
except ModuleNotFoundError:
    HAS_CCTBX=False



def Fs_from_pdb(pdb_file):
    pdb_in = iotbx_pdb.input(pdb_file)
    xray_structure = pdb_in.xray_structure_simple(enable_scattering_type_unknown=True)
    scatts = xray_structure.scatterers()
    xray_structure.erase_scatterers()
    for sc in scatts:
        try:
            _ = sc.electron_count()  # fails for unknown scatterer types
        except RuntimeError:
            continue
        xray_structure.add_scatterer(sc)

    fcalc = xray_structure.structure_factors(
        d_min=2,
        algorithm='fft',
        anomalous_flag=True)
    F = fcalc.f_calc().as_amplitude_array()
    return F

def _prep_miller_array(mill_arr):
    """prep for nanoBragg"""
    # TODO: is this correct order of things?
    cb_op = mill_arr.space_group_info().change_of_basis_op_to_primitive_setting()
    mill_arr = mill_arr.expand_to_p1()
    mill_arr = mill_arr.generate_bijvoet_mates()
    dtrm = cb_op.c().r().determinant()
    if not dtrm == 1:
        mill_arr = mill_arr.change_basis(cb_op)
    return mill_arr


def batch_cross(X, Y):
    """
    :param X: tensor Nx3
    :param Y: tensor Nx3
    :return: tensor Nx3
    """
    #x0 = X[:,0]
    #x1 = X[:,1]
    #x2 = X[:,2]

    #y0 = Y[:,0]
    #y1 = Y[:,1]
    #y2 = Y[:,2]
    z0 = X[:,1]*Y[:,2] - X[:,2]*Y[:,1]
    z1 = X[:,2]*Y[:,0] - X[:,0]*Y[:,2]
    z2 = X[:,0]*Y[:,1] - X[:,1]*Y[:,0]

    #z0 = x1*y2 - x2*y1
    #z1 = x2*y0 - x0*y2
    #z2 = x0*y1 - x1*y0
    Z = torch.vstack((z0, z1, z2)).transpose(1,0)
    return Z


def debug(M,a):
    if np.any( np.isnan(M.cpu().detach().numpy().flatten() )):
        print("NAN in :", a)


def gs_mapping(vecs):
    """
    :param vecs: outputs of the ori_mode resnet model (6 parameters per image)
    :return: 9 parameters per image (the orientation matrix)
    """
    a1 = vecs[:, 0:3]
    a2 = vecs[:, 3:6]
    a1_norm = torch.linalg.norm(a1, axis=1).clamp(1e-6)
    b1 = a1 / a1_norm[:,None]

    b1_dot_a2 = torch.sum(b1 * a2, axis=1)
    u2 = a2 - b1_dot_a2[:,None] * b1

    u2_norm = torch.linalg.norm(u2,axis=1).clamp(min=1e-6)
    b2 = u2 / u2_norm[:,None]

    b3 = batch_cross(b1, b2)
    #b3 = torch.cross( b1, b2)

    rot_mat_elems = torch.hstack((b1, b2, b3))

    #rot_mat_elems = []
    #for vec in vecs:
    #    a1 = vec[0::2]
    #    a2 = vec[1::2]
    #    a1_norm = torch.linalg.norm(a1)
    #    b1 = a1 / a1_norm
    #    u2 = a2 - torch.dot(b1, a2)*b1
    #    b2 = u2 / torch.linalg.norm(u2)
    #    b3 = torch.cross(b1, b2)
    #    rot_mat = torch.vstack((b1, b2, b3)).T
    #    rot_mat_elems.append(  torch.hstack((b1,b2,b3)))

    return rot_mat_elems.view(-1,3,3).transpose(1,2)


def loss(model_rots, gt_rots, reduce=True, sgnums=None):
    """
    Below, `N` stands for batch size
    :param model_rots: output of the ori_mode=True model (N x 9) tensor
    :param gt_rots: ground truth orientations (N x 9) tensor
    :param reduce: whether to return the summed loss, or one per example
    :return:
    """
    mat_prod = torch.bmm(model_rots, gt_rots.reshape((-1, 3, 3)).transpose(1, 2))
    diags = torch.diagonal(mat_prod, dim1=2, dim2=1)
    traces = .5*diags.sum(1)-.5
    loss = torch.arccos(torch.clamp( traces, -1+1e-6, 1-1e-6))
    if reduce:
        loss = loss.mean()
    return loss


class Loss(torch.nn.Module):
    def __init__(self, sgop_table, pdb_id_to_num, dev, *args, **kwargs):
        """
        :param sgop_table:
        :param pdb_id_to_num:
        """
        super().__init__(*args, **kwargs)
        self.Nop = max([len(v) for v in sgop_table.values()])
        self.Nsym = len(sgop_table)
        self.sgops = np.zeros((self.Nsym, self.Nop, 3, 3))
        self.sgops[:, :] = np.eye(3)  # default is the Identity
        for pdb_id in sgop_table:
            rots = sgop_table[pdb_id]
            idx = pdb_id_to_num[pdb_id]
            self.sgops[idx, :len(rots)] = rots
        self.sgops = torch.tensor(self.sgops.astype(np.float32), device=dev)

    def forward(self, model_rots, gt_rots, reduce=True, sgnums=None):
        """
        Below, `N` stands for batch size
        :param model_rots: output of the ori_mode=True model (N x 9) tensor
        :param gt_rots: ground truth orientations (N x 9) tensor
        :param reduce: whether to return the summed loss, or one per example
        :return:
        """
        if sgnums is not None:
            # sgnums is list of ints e.g. 0, 1, 1, 2, 4, 24, ..
            # model_rots is Nbatch x  3x  3
            # rots is Nbatch x Nop x 3 x 3
            #
            rots = self.sgops[sgnums]
            model_rots_op = torch.matmul(model_rots[:,None], rots)
            G = gt_rots.reshape((-1,3,3)).transpose(1,2)
            mat_prod = torch.matmul(model_rots_op, G[:,None])
            diags = torch.diagonal(mat_prod, dim1=3, dim2=2)
            traces = .5 * diags.sum(2) - .5
            loss = torch.arccos(torch.clamp(traces, -1 + 1e-6, 1 - 1e-6))
            loss = loss.min(1).values

            #rots = self.sgop_table[sgnums]  # Nbatch x Nops x 3 x 3
            #for i_op in range(len(rots)):
            #    model_rots_op = torch.matmul(rots[:, i_op], model_rots)  # Nbatch x 3 x 3
            #    mat_prod = torch.bmm(model_rots_op, gt_rots.reshape((-1, 3, 3)).transpose(1, 2))
            #    diags = torch.diagonal(mat_prod, dim1=2, dim2=1)
            #    traces = .5*diags.sum(1)-.5
            #    loss_i = torch.arccos(torch.clamp(traces, -1+1e-6, 1-1e-6))
            #    if i_op == 0:
            #        loss = loss_i
            #    else:
            #        loss = torch.minimum(loss, loss_i)

        else:
            mat_prod = torch.bmm(model_rots, gt_rots.reshape((-1, 3, 3)).transpose(1, 2))
            diags = torch.diagonal(mat_prod, dim1=2, dim2=1)
            traces = .5*diags.sum(1)-.5
            loss = torch.arccos(torch.clamp(traces, -1+1e-6, 1-1e-6))

        if reduce:
            loss = loss.mean()
        return loss


def make_op_table_using_nanoBragg(outfile, cuda=True):
    assert HAS_CCTBX
    pdb_path = os.path.join(os.environ["RESONET_SIMDATA"], "pdbs")
    pdb_id_file = os.path.join(pdb_path, "pdb_ids.txt")
    pdb_ids = [p.strip() for p in open(pdb_id_file, "r").readlines()]

    pdb_ops = {}
    for i_pdb, pdb_id in enumerate(pdb_ids):
        print("Beginngin PDB %s (%d / %d)" % (pdb_id, i_pdb+1, len(pdb_ids)))

        add_spots_func = "add_nanoBragg_spots"
        if cuda:
            add_spots_func = "add_nanoBragg_spots_cuda"
        os.system("iotbx.fetch_pdb %s" % pdb_id)
        pdb_file = pdb_id + ".pdb"
        assert os.path.exists(pdb_file)
        print("Downloaded %s." % pdb_file)
        F = Fs_from_pdb(pdb_file)

        sg = F.space_group()
        sgi = sg.info()
        print("PDB unit cell, space group:\n", F, "\n")

        ucell = F.crystal_symmetry().unit_cell()
        O = ucell.orthogonalization_matrix()
        # real space vectors
        a = O[0], O[3], O[6]
        b = O[1], O[4], O[7]
        c = O[2], O[5], O[8]
        C_sg = Crystal(a, b, c, sg)
        B_sg = np.reshape(C_sg.get_B(), (3,3))
        # this should have the identity as a Umat
        assert np.allclose(C_sg.get_U(), (1, 0, 0, 0, 1, 0, 0, 0, 1))

        to_p1_op = sgi.change_of_basis_op_to_primitive_setting()
        # this block of code is used to compute the "phantom Umat" thats introduced when
        # converting certain space groups to P1
        # to p1 matrix, these should be identical:
        Oi_test = np.linalg.inv(np.reshape(to_p1_op.c_inv().r().transpose().as_double(), (3, 3)))
        Oi = np.reshape(to_p1_op.c().r().transpose().as_double(), (3, 3))
        assert np.allclose(Oi, Oi_test)

        # real space B-matrix in P1
        Breal_p1 = np.linalg.inv(B_sg @ Oi).T
        # convert Breal to upper triangular representation
        uc = uctbx.unit_cell(orthogonalization_matrix=tuple(Breal_p1.ravel()))
        Breal_p1 = np.reshape(uc.orthogonalization_matrix(), (3, 3))
        Brecip_p1 = np.linalg.inv(Breal_p1).T
        Bi = np.linalg.inv(Brecip_p1)
        # phantom Umat:
        U_p = B_sg @ Oi @ Bi

        # convert crystal to P1
        C_p1 = C_sg.change_basis(to_p1_op)
        # note, the Umat of C_p1 should be U_p, and not always identity
        assert np.allclose( C_p1.get_U(), U_p.ravel())

        # Generate a diffraction pattern at a random orientation
        # random orientation
        Misset = sqr(Rotation.random(random_state=1).as_matrix().flatten())
        # reorient the P1 crystal
        U_p1 = sqr(C_p1.get_U())
        C_p1.set_U(Misset * U_p1)

        # make a detector and a beam
        beam = BeamFactory.simple(wavelength=1)
        detector = DetectorFactory.simple(
            sensor='PAD',
            distance=200,
            beam_centre=(51.25, 51.25),
            fast_direction='+x',
            slow_direction='-y',
            pixel_size=(.1, .1),
            image_size=(1024, 1024))

        SIM = nanoBragg(detector=detector, beam=beam)
        SIM.oversample = 1
        SIM.interpolate = 0
        # convert to P1 (and change basis depending on space group)
        F_p1 = _prep_miller_array(F)
        SIM.Fhkl_tuple = F_p1.indices(), F_p1.data()
        SIM.Amatrix = sqr(C_p1.get_A()).transpose()
        ucell_p1 = C_p1.get_unit_cell().parameters()
        assert np.allclose(SIM.unit_cell_tuple, ucell_p1)
        SIM.Ncells_abc = 7, 7, 7
        getattr(SIM, add_spots_func)()
        # this is the reference image which we will compare to below after rotating the crystal and re-simulating
        reference = SIM.raw_pixels.as_numpy_array()

        #sanity_check_Fs(SIM, sg, C_p1.get_unit_cell().parameters())

        # list all of the laue group operators, should have no translations
        ops = sg.build_derived_laue_group().all_ops()
        ops = [o for o in ops if o.r().determinant() == 1]
        Umat_ops = []
        for o in ops:
            assert o.t().is_zero()

            # we perturb the crystal model and re-simulate
            sg_op = sgtbx.change_of_basis_op(o)
            C_o = C_sg.change_basis(sg_op).change_basis(to_p1_op)
            U_o = sqr(C_o.get_U())
            C_o.set_U(Misset * U_o)
            Amat = sqr(C_o.get_A())
            SIM.raw_pixels *= 0  # reset the image
            SIM.Amatrix = Amat.transpose()
            getattr(SIM, add_spots_func)()
            # new image, should be equivalent to reference if the symop preserved the diffraction
            img = SIM.raw_pixels.as_numpy_array()

            ucell = [round(x, 2) for x in SIM.unit_cell_tuple]
            print("operator:", o.as_xyz(), "ucell:", ucell)
            forms = literal_description(o)
            print(forms.long_form())
            if not np.allclose(img, reference):
                print("Image comparison failed\n")
            else:
                # in practice, if we know a patterns U-mat, and we want to test an equivalent, then we should
                # right multiply the patterns U-mat by this matrix U_m:
                U_m = np.reshape(o.r().as_double(), (3,3))
                U_m = B_sg @ U_m.T @ np.linalg.inv(B_sg)
                U_m = np.linalg.inv(U_p) @ U_m @ U_p
                Umat_ops.append(U_m)
                print("Images are identical.\n")

        print(str(sgi).replace(" ", ""), ": %d/%d Umats saved" % (len(Umat_ops), len(ops)))
        SIM.free_all()
        pdb_ops[pdb_id] = Umat_ops

    np.save(outfile, pdb_ops)
    return pdb_ops


def make_op_table_old(outfile):
    """
    NOTE: this doesnt quite work for some space groups (P3, P6, I, F)
    Creates lists of rotation operators for each PDBfile simulated that should presever the diffraction patern
    these are to be used for training loss calculation
    :param outfile: file to save , to be assigned to resonet.sims.paths_and_const.SGOP_FILE
    """
    pdb_path = os.path.join(os.environ["RESONET_SIMDATA"], "pdbs")
    pdb_id_file = os.path.join(pdb_path, "pdb_ids.txt")
    pdb_ids = [p.strip() for p in open(pdb_id_file, "r").readlines()]

    pdb_ops = {}
    for p in pdb_ids:
        pdb_file = os.path.join(pdb_path, "%s/%s.pdb" %(p,p))
        P = process_pdb.PDB(pdb_file)

        # change of basis to p1 operator
        O = np.reshape(P.to_p1_op.c_inv().r().transpose().as_double(), (3, 3))
        Oi = np.linalg.inv(O)

        # recip space B-matrix in reference setting
        Bc = np.reshape(P.dxtbx_crystal.get_B(), (3,3))
        # real space B-matrix in P1
        Breal = np.linalg.inv(Bc @ Oi).T

        # convert Breal to upper triangular representation
        uc = uctbx.unit_cell(orthogonalization_matrix=tuple(Breal.ravel()))
        Breal = np.reshape(uc.orthogonalization_matrix(), (3,3))

        # phantom Umat that is sometimes introduced when expanding to P1
        Up = Bc @ Oi @ Breal.T

        # loop over space group operations that preserve the diffraction pattern
        sg = P.sym.space_group()
        ops = sg.build_derived_laue_group().all_ops()
        # keep the right-handed operators
        ops = [o for o in ops if o.r().determinant() == 1]
        # check translation like dials/algorithms/indexing/compare_orientation_matrices?
        #assert all([o.t().is_zero() for o in ops])

        Corig = deepcopy(P.dxtbx_crystal)
        pdb_ops[p] = []
        for o in ops:

            # should this b o.inverse like dials/algorithms/indexing/compare_orientation_matrices ?
            o = sgtbx.change_of_basis_op(o)

            Corig_o = Corig.change_basis(o)
            C_p1 = Corig_o.change_basis(P.to_p1_op)
            U = np.reshape(C_p1.get_U(), (3, 3))
            U = Up.T @ U
            # Now, if a crystal U-matrix is multiplied by this U, then the diffraction pattern should remain unchanged
            pdb_ops[p].append(U)

    np.save(outfile, pdb_ops)
    return pdb_ops