
import numpy as np
import torch
import os
from copy import deepcopy
from cctbx import sgtbx, uctbx


from resonet.sims import process_pdb


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


def make_op_table(outfile):
    """
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