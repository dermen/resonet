
import glob
import os

"""
Some constants and file paths for simulations
"""

# this file contains the P1 amplitudes used to simulate diffraction
# these files are slow to read: TODO convert and store in high symmetry and use CCTBX
# to convert to P1

dirname = os.path.join(os.path.dirname(__file__), "for_tutorial/diffraction_ai_sims_data")
if not os.path.exists(dirname):
    print("Warning, Could not locate simulation data. Simulations wont work until running `resonet-getsimdata`.")
    dirname="."

# these are scattering profiles for random plastics (from James Holton)
RANDOM_STOLS = glob.glob(os.path.join(dirname, "randomstols/*stol"))
#RANDOM_STOLS = [r for r in RANDOM_STOLS if "water_014" in r]
# scattering profiles for air and water
AIR_STOL = os.path.join(dirname, "air.stol")
WATER_STOL = os.path.join(dirname, "water.stol")
#STOL_MIN = 0.025
STOL_MIN = 0.15
STOL_MAX = 0.35
STOL_RNG = STOL_MAX-STOL_MIN

DIVERGENCE_MRAD = 0
DIVERGENCE_NSTEPS = 0
LAMBDA_FILE = os.path.join(dirname, "e080_2.lam")
LAUE_MODE = False
PEAK_MODE = False
BEAM_SIZE_MM = 0.03
FLUX = 4e11  # photons per pulse
XTALSIZE_MM = 0.025
#CENTER_WINDOW_MM = 100  # if randomizing beam center, vary the center around a box of this edge size
CENTER_WINDOW_MM = 3  # if randomizing beam center, vary the center around a box of this edge size
#DOMAINSIZE_MM = 30e-5
#DOMAINSIZE_MM = 15e-5
DOMAINSIZE_MM = 5e-5
FLAT_BACKGROUND = False
VOL = (XTALSIZE_MM / DOMAINSIZE_MM)**3  # scales the diffraction
FIX_RES = None # 0.5/.29989  # optionally fix the resolution for all simulations ...
RANDOM_TILT_PITCH_DEG = 3
RANDOM_TILT_YAW_DEG = 3

CUT_1P2 = True # try loading the 1p2 fmodel files (assuming they were created). This is simply the original fmodel files cut at 1.2 Angstrom, and should significantly speed up throughput

# these are the PDB folders containing pdb files and P1.hkl files
RANDOM_PDBS = [d for d in glob.glob(os.path.join(dirname, "pdbs/*")) if len(os.path.basename(d))==4 and os.path.isdir(d)]

# these have possible twin laws: "5v5k", "3int", "1h74"
#whitelist = ["1hk5", "2pkg",
#             "3nxs", "4fhm", "1z35", "3t4x"]
#whitelist = ["1nne"]#, "3k6n"]

# no twinning laws:
#whitelist = ['1hk5', '1keq', '1ktc', '1lbv', '1nne', '1qtx', '1r03', '1sg8', '1uic', '1vh6',
#             '1xrt', '1yj1', '1yo6', '1z35', '2ar6', '2bh4', '2cc3', '2hu3', '2hyf', '2i8d',
#             '2ibm', '2pkg', '2qa4', '2qex', '2wyf', '2x8i', '2y8k', '2zg2', '2znt', '2zry',
#             '3agy', '3ch7', '3cma', '3cpw', '3dll', '3e6l', '3fj8', '3g8y', '3hxf', '3ilo',
#             '3k6n', '3lke', '3nxs', '3t4x', '3tuu', '3uhr', '3woz', '3zbs', '4cbc', '4ctn',
#             '4dvn', '4fhm', '4j20', '4m97', '4o09', '4o7s', '4p9h', '4pgu', '4xbe', '4xxo',
#             '4z40', '5al4', '5avi', '5dt6', '5g4e', '5jit', '5p9i', '5pjt', '5v5v', '5wqg',
#             '6csc']
##whitelist = ["4pgu"] # P6
##whitelist = ["1r03"] # F
#
## sym ops sanity test fails for these unless using Gaussian Star model:
##failed = ['1r03', '2ibm', '2zg2', '3dll', '4o09', '5pjt']
#
## twinning laws
#blacklist = ['1h74', '1pdv', '1rlk', '1uv7', '1z6s', '2itu', '2nrz', '2qma', '2qt4', '2vj3',
#             '2vuy', '2wox', '2xh6', '2ycf', '3dxj', '3fl2', '3fyx', '3hfp', '3int', '3l89',
#             '3lz7', '3n0w', '3oj1', '3u7s', '3uh4', '3vgd', '3wpz', '3zg2', '4arq', '4e6i',
#             '4f3x', '4gyk', '4m5i', '4px8', '4qxq', '4rmx', '4wd2', '4ypu', '5aoo', '5g52',
#             '5j77', '5o99', '5v5k', '5vn7', '5vn9', '5xg2']
## big viruses and ribosomal units
#blacklist += ['3dll', '5g52', '5aoo', '2qex', '2qa4', '3cpw', '3cma', '3dxj']
#
## test set reserved!
#blacklist += ['1r03', '1hk5', '4fhm', '3u7s']
##whitelist = ['1r03', '1hk5', '4fhm', '3u7s']
#
#RANDOM_PDBS = [d for d in RANDOM_PDBS if any([p in d for p in whitelist]) and not any(p in d for p in blacklist)]
##RANDOM_PDBS = [RANDOM_PDBS[0]]
#
PDB_MAP = {name: i for i, name in enumerate(RANDOM_PDBS)}
SGOP_FILE = os.path.join(dirname, "pdb_ops.npy")

# mosaicity bounds (degrees)
MOS_MIN = 0.2
MOS_MAX = 1

