
import os
import sys
from iotbx import pdb as iotbx_pdb
from simtbx.nanoBragg import nanoBragg
from scitbx.matrix import sqr
from dxtbx.model import Crystal
from scipy.spatial.transform import Rotation
import numpy as np

from dxtbx.model import DetectorFactory, BeamFactory
from cctbx import sgtbx, miller, crystal
from cctbx.array_family import flex


# ===================
# prepare the miller array for nanoBragg
def miller_array(val):
    cb_op = val.space_group_info().change_of_basis_op_to_primitive_setting()
    val = val.expand_to_p1()
    val = val.generate_bijvoet_mates()
    dtrm = cb_op.c().r().determinant()
    if not dtrm == 1:
        val = val.change_basis(cb_op)
    return val


# simple method to check whether the structure factors in nanoBragg obey the symmetry
def sanity_check_Fs(SIM, sg, ucell_p1):
    """
    SIM: nanoBragg instance
    sg: space group operator
    ucell_p1: p1 unit cell params
    """
    inds, amps = SIM.Fhkl_tuple
    print(ucell_p1)

    sym = crystal.symmetry(ucell_p1, "P1")
    mset = miller.set(sym, flex.miller_index(inds), True)
    ma = miller.array(mset, flex.double(amps)).set_observation_type_xray_amplitude().resolution_filter(-1,
                                                                                                       2)
    print(sg.info())
    ops = [o for o in sg.build_derived_laue_group().all_ops() if o.r().determinant() == 1]
    for o in ops:
        print("Op=", o.as_xyz())
        ma2 = ma.change_basis(sgtbx.change_of_basis_op(o))
        r = ma.r1_factor(ma2, assume_index_matching=True)
        print("R=", r, "\n")
        assert r == 0
# ===================


def test(pdb_id, cuda=False):
    add_spots_func="add_nanoBragg_spots"
    if cuda:
        add_spots_func = "add_nanoBragg_spots_cuda"
    os.system("iotbx.fetch_pdb %s" % pdb_id)
    pdb_file = pdb_id+".pdb"
    assert os.path.exists(pdb_file)
    print("Downloaded %s."%pdb_file)
    pdb_in = iotbx_pdb.input(pdb_file)
    xray_structure = pdb_in.xray_structure_simple()
    fcalc = xray_structure.structure_factors(
        d_min=2,
        algorithm='fft',
        anomalous_flag=True)
    F = fcalc.f_calc().as_amplitude_array()
    sg = F.space_group()
    sgi = sg.info()
    print("PDB unit cell, space group:\n", F, "\n")

    ucell = pdb_in.crystal_symmetry().unit_cell()
    O = ucell.orthogonalization_matrix()
    # real space vectors
    a = O[0], O[3], O[6]
    b = O[1], O[4], O[7]
    c = O[2], O[5], O[8]
    C_sg = Crystal(a,b,c,sg)
    # this should have the identity as a Umat
    assert np.allclose(C_sg.get_U(), (1,0,0,0,1,0,0,0,1))

    # convert crystal to P1
    to_p1_op = sgi.change_of_basis_op_to_primitive_setting()
    C_p1 = C_sg.change_basis(to_p1_op)

    # random orientation
    Misset = sqr(Rotation.random(random_state=1).as_matrix().flatten())
    # reorient the P1 crystal
    U_p1 = sqr(C_p1.get_U())
    C_p1.set_U(Misset*U_p1)

    beam = BeamFactory.simple(wavelength=1)
    detector =DetectorFactory.simple(
        sensor='PAD',
        distance=200,
        beam_centre=(51.25,51.25),
        fast_direction='+x',
        slow_direction='-y',
        pixel_size=(.1,.1),
        image_size=(1024,1024))

    SIM = nanoBragg(detector=detector, beam=beam)
    SIM.oversample = 1
    SIM.interpolate = 0
    F_p1 = miller_array(F)
    SIM.Fhkl_tuple = F_p1.indices(), F_p1.data()
    SIM.Amatrix = sqr(C_p1.get_A()).transpose()
    assert np.allclose(SIM.unit_cell_tuple, C_p1.get_unit_cell().parameters())
    SIM.Ncells_abc = 7, 7, 7
    getattr(SIM, add_spots_func)()
    reference = SIM.raw_pixels.as_numpy_array()

    sanity_check_Fs(SIM, sg, C_p1.get_unit_cell().parameters())

    ops = sg.build_derived_laue_group().all_ops()
    ops = [o for o in ops if o.r().determinant() == 1]

    failed_imgs = []
    for o in ops:
        print("operator:", o.as_xyz())
        sg_op = sgtbx.change_of_basis_op(o)
        C_o = C_sg.change_basis(sg_op).change_basis(to_p1_op)
        U = sqr(C_o.get_U())
        C_o.set_U(Misset*U)
        Amat = sqr(C_o.get_A())
        SIM.raw_pixels *= 0
        SIM.Amatrix = Amat.transpose()
        getattr(SIM, add_spots_func)()
        img = SIM.raw_pixels.as_numpy_array()
        if not np.allclose(img, reference):
            print("Image comparison failed\n")
            failed_imgs.append(img)
        else:
            print("Images are identical.\n")

    SIM.free_all()
    return failed_imgs, sgi

if __name__=="__main__":
    cuda = True
    if len(sys.argv)==2:
        pdb_id = sys.argv[1]
        failed_imgs, _ = test(pdb_id, cuda)
        assert not failed_imgs
        exit()

    #pdb_id_file = os.path.join(os.environ["RESONET_SIMDATA"], "pdbs/pdb_ids.txt")
    #pdb_ids = [l.strip() for l in open(pdb_id_file, "r").readlines()]
    #passes = {}
    #fails = {}
    #for pdb_id in pdb_ids:
    #    try:
    #        failed_imgs, sgi = test(pdb_id, cuda)
    #        if failed_imgs:
    #            fails[pdb_id] = sgi.type().lookup_symbol()
    #        else:
    #            passes[pdb_id] = sgi.type().lookup_symbol()
    #    except:
    #        continue
    #from IPython import embed;embed()