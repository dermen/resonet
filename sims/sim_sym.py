
import os
import sys
from scipy.spatial.transform import Rotation
import numpy as np

from iotbx import pdb as iotbx_pdb
from simtbx.nanoBragg import nanoBragg
from scitbx.matrix import sqr
from dxtbx.model import Crystal
from dxtbx.model import DetectorFactory, BeamFactory
from cctbx import sgtbx, miller, crystal
from cctbx.array_family import flex

"""
Usage (the script will download the PDB code provided on command line):

  libtbx.python sim_sym.py 4bs7
  
To show plots add `plot` to the command line:
  
  libtbx.python sim_sym.py 4bs7 plot
  
The script attempts to simulate diffraction patterns
for symmetrically equivalent crystal orientations, and tests if the intensities change. 
If changes in intensity are detected, the stdout will indicate Failure.

So far, the tests pass for these space groups:
  C2221, P21212, P41212, P42212, P1, R32:H, R3:H, 
  P4132, C121, P41, P43212, P212121, P1211, P4212, P213

***The tests Fail for these space groups:
  I222, P6522, P32, P321, P6, I23, P3221, 
  P6122, P3, P61, I212121, P3121, F432, P3112
  
PDB codes to test for each space group:
  C121 (1hk5), C2221 (2pkg), F432 (1r03), I212121 (2ibm), 
  I222 (3dll), I23 (2itu), P1 (3l89), P1211 (1h74), 
  P21212 (3nxs), P212121 (1nne), P213 (5v5k), P3 (4rmx), 
  P3112 (4wd2), P3121 (1pdv), P32 (3dxj), P321 (2vuy), P3221 (1uv7), 
  P41 (3int), P41212 (4fhm), P4132 (1z35), P4212 (3k6n), P42212 (3t4x), 
  P43212 (1ktc), P6 (4qxq), P61 (3u7s), P6122 (3e6l), P6522 (4pgu), 
  R32:H (2zg2), R3:H (5vn7)

"""

# ===================
# prepare the miller array for nanoBragg
def prep_miller_array(mill_arr):
    """prep for nanoBragg"""
    # TODO: is this correct order of things?
    cb_op = mill_arr.space_group_info().change_of_basis_op_to_primitive_setting()
    mill_arr = mill_arr.expand_to_p1()
    mill_arr = mill_arr.generate_bijvoet_mates()
    dtrm = cb_op.c().r().determinant()
    if not dtrm == 1:
        mill_arr = mill_arr.change_basis(cb_op)
    return mill_arr


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

cuda = True
pdb_id = sys.argv[1]

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

# make a detector and a beam
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
# convert to P1 (and change basis depending on space group)
F_p1 = prep_miller_array(F)
SIM.Fhkl_tuple = F_p1.indices(), F_p1.data()
SIM.Amatrix = sqr(C_p1.get_A()).transpose()
assert np.allclose(SIM.unit_cell_tuple, C_p1.get_unit_cell().parameters())
SIM.Ncells_abc = 7, 7, 7
getattr(SIM, add_spots_func)()
reference = SIM.raw_pixels.as_numpy_array()

sanity_check_Fs(SIM, sg, C_p1.get_unit_cell().parameters())

ops = sg.build_derived_laue_group().all_ops()
ops = [o for o in ops if o.r().determinant() == 1]

results = []
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
        passed=False
        print("Image comparison failed\n")
    else:
        passed=True
        print("Images are identical.\n")
    results.append((passed, img, o))

SIM.free_all()
nfail = sum([not passed for passed,_,_ in results ])
nops = len(results)
if nfail > 0:
    print("Test failed for %d / %d ops." %(nfail, len(results)))
else:
    print("All tests pass (%d ops) ! " %(len(results)))
    print("Ok!")

if "plot" in sys.argv:
    try:
        from pylab import *
        from itertools import cycle
        imgs = cycle(results)
        m = reference[reference >0].mean()
        s = reference[reference>0].std()
        vmax=m+0.5*s
        fig, (ax1,ax2) = subplots(nrows=1, ncols=2, layout='constrained')
        fig.set_size_inches((8,4))
        ax1.imshow(reference, vmin=0, vmax=vmax)
        ax2.imshow(results[0][1], vmin=0, vmax=vmax)
        ax1.set_title("x,y,z", fontsize=16)
        suptitle("%s (%s)" % (pdb_id, str(sgi)), fontsize=16)
        while 1:
            if not fignum_exists(fig.number):
                break
            passed, img, op = next(imgs)
            ax2.images[0].set_data(img)
            pass_s = "passed" if passed else "failed"
            ax2.set_title("%s (%s)" % (op.as_xyz(), pass_s), fontsize=16)
            plt.draw()
            plt.pause(2)

    except KeyboardInterrupt:
        close()
        exit()