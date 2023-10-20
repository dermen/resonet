
import glob
import os

"""
Some constants and file paths for simulations
"""

# this file contains the P1 amplitudes used to simulate diffraction
# these files are slow to read: TODO convert and store in high symmetry and use CCTBX
# to convert to P1

dirname = os.environ.get("RESONET_SIMDATA")
if dirname is None:
    print("Warning, RESONET_SIMDATA is not set, simulation might not work!")
    dirname="."

# these are scattering profiles for random plastics (from James Holton)
RANDOM_STOLS = glob.glob(os.path.join(dirname, "randomstols/*stol"))
# scattering profiles for air and water
AIR_STOL = os.path.join(dirname, "air.stol")
WATER_STOL = os.path.join(dirname, "water.stol")
STOL_MIN = 0.025
STOL_MAX = 0.35
STOL_RNG = STOL_MAX-STOL_MIN

BEAM_SIZE_MM = 0.03
FLUX = 4e11  # photons per pulse
DIVERGENCE_MRAD = 0.02
XTALSIZE_MM = 0.025
DOMAINSIZE_MM = 5e-5
VOL = (XTALSIZE_MM / DOMAINSIZE_MM)**3  # scales the diffraction
FIX_RES = 1.5 #None # 0.5/.29989  # optionally fix the resolution for all simulations ...

CUT_1P2 = False  # try loading the 1p2 fmodel files (assuming they were created). This is simply the original fmodel files cut at 1.2 Angstrom, and should significantly speed up throughput

# these are the PDB folders containing pdb files and P1.hkl files
RANDOM_PDBS = [d for d in glob.glob(os.path.join(dirname, "pdbs/*")) if len(os.path.basename(d))==4 and os.path.isdir(d)]
PDB_MAP = {name: i for i, name in enumerate(RANDOM_PDBS)}
SGOP_FILE = os.path.join(dirname, "ops_info_96.npz")

# mosaicity bounds (degrees)
MOS_MIN = 0.2
MOS_MAX = 1

