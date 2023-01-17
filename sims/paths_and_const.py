
import glob
import os
import math

"""
Some constants and file paths for simulations
"""

# this file contains the P1 amplitudes used to simulate diffraction
# these files are slow to read: TODO convert and store in high symmetry and use CCTBX
# to convert to P1
P1_FILE = "/data/dermen/sims/pdb_P1_hkls.h5"
FILE = "/data/dermen/sims/pdb_highSym_hkls.h5"


# these are scattering profiles for random plastics (from James Holton)
RANDOM_STOLS = glob.glob("/data/dermen/sims/randomstols/*stol")
# scattering profiles for air and water
AIR_STOL = "/data/dermen/sims/air.stol"
WATER_STOL = "/data/dermen/sims/water.stol"
STOL_MIN = 0.025
STOL_MAX = 0.35
STOL_RNG = STOL_MAX-STOL_MIN

BEAM_SIZE_MM = 0.03
FLUX = 1e12  # photons per pulse
DIVERGENCE_MRAD = 0.02
XTALSIZE_MM = 0.025
DOMAINSIZE_MM = 5e-5
VOL = (XTALSIZE_MM / DOMAINSIZE_MM)**2  # scales the diffraction

# these are the PDB folders containing pdb files and P1.hkl files
RANDOM_PDBS = [d for d in glob.glob("/data/dermen/sims/pdbs/*") if len(os.path.basename(d))==4 and os.path.isdir(d)]

# minimum mosaicity
MOS_MIN = 0.2
MOS_RNG = (math.sqrt(1) - math.sqrt(.2))**2
