
import os
import process_pdb
from cctbx.array_family import flex
from cctbx import miller, crystal
from simtbx.nanoBragg import nanoBragg_crystal, nanoBragg_beam
import numpy as np
import paths_and_const
import h5py


def convert_inds(amps, hkls, pdb_file, d_min=1.2):
    """
    :param amps: data as doubles
    :param hkls: indices
    :param pdb_file:
    :param d_min:
    :return:
    """
    P = process_pdb.PDB(pdb_file)
    midx = flex.miller_index(hkls)
    amps = flex.double(amps)
    sym = crystal.symmetry(P.p1_ucell, "P1")
    mset = miller.set(sym, midx, True)
    #op = P.sym.change_of_basis_op_to_primitive_setting().inverse()
    op = P.to_p1_op.inverse()
    mset2 = miller.set(P.sym, mset.change_basis(op).indices(), True)
    ma = miller.array(mset2, amps)
    ma = ma.set_observation_type_xray_amplitude()
    ma = ma.resolution_filter(d_min=d_min).merge_equivalents().array()
    new_inds = np.array(list(ma.indices()))
    new_amps = ma.data().as_numpy_array()
    return new_amps, new_inds


def load_amps(hkl_file, ucell, sg="P1", use_hdf5=True):
    """
    :param hkl_file: 4-column text files, P1 miller indices (cols 1-3) and amplitudes (col 4)
    :param ucell: unit cell dimensions (if sg=P1, ensure ucell is p1 equivalent cell)
    :param sg: space group lookup symbol , e.g. P43212
    :param use_hdf5: use the hdf5 file
    :return: cctbx miller array object
    """
    if use_hdf5:
        fname = paths_and_const.P1_FILE if sg=="P1" else paths_and_const.FILE
        with h5py.File(fname, "r") as H:
            pdb_id = os.path.basename(os.path.dirname(hkl_file))
            amps = H[pdb_id]["amps"][()]
            if not amps.flags.contiguous:
                amps = np.ascontiguousarray(amps)
            amps = flex.double(amps)

            hkl = H[pdb_id]["hkl"][()]
            if not hkl.dtype==np.int32:
                hkl = hkl.astype(np.int32)
            midx = flex.miller_index(hkl)
    else:
        hklF = np.loadtxt(hkl_file)
        midx = flex.miller_index(hklF[:,:3].astype(np.int32))
        amps = flex.double(np.ascontiguousarray(hklF[:,3]))

    sym = crystal.symmetry(ucell, sg)
    mset = miller.set(sym, midx, True)
    ma = miller.array(mset, amps)
    return ma


def get_Nabc(ucell):
    """

    :param ucell: unit cell dimension
    :return: number of unit cells along each unit cell axis, according to domain size of crystal
            see paths_and_const.DOMAINSIZE_MM
    """
    size_m = paths_and_const.DOMAINSIZE_MM*1e-3
    Na = np.ceil(size_m / (ucell[0]*1e-10))
    Nb = np.ceil(size_m / (ucell[1]*1e-10))
    Nc = np.ceil(size_m / (ucell[2]*1e-10))
    Na = max(Na, 3)
    Nb = max(Nb, 3)
    Nc = max(Nc, 3)
    return Na, Nb, Nc


def load_crystal(folder, rot_mat=None, use_p1=False):
    """

    :param folder:  pdb folder, e.g. /data/dermen/sims/pdbs/2itu
        each folder basename 2itu should also exist in the paths_and_const.P1_FILE hdf5 file if reading from hdf5
        If not, the files ./pdbs/2itu/2itu.pdb are and ./pdbs/2itu/P1.hkl are expected to exist
    :param rot_mat: rotation matrix of crystal (otherwise it will be aligned in the standard PDB convention)
    :param use_p1: boolean to use p1 space group
    :return:
    """
    C = nanoBragg_crystal.NBcrystal(init_defaults=False)
    assert os.path.isdir(folder)
    pdb_base = os.path.basename(folder)
    P = process_pdb.PDB(folder + '/%s.pdb' % pdb_base)
    C.dxtbx_crystal = P.p1_dxtbx_crystal
    C.xtal_shape = "tophat"
    if rot_mat is not None:
        Umat = C.dxtbx_crystal.get_U()
        Umat = np.dot(rot_mat, np.reshape(Umat,(3,3)))
        C.dxtbx_crystal.set_U(tuple(Umat.ravel()))
    C.Ncells_abc = get_Nabc(P.p1_ucell)
    if use_p1:
        ma = load_amps(folder + "/P1.hkl", P.p1_ucell, sg="P1")
    else:
        ma = load_amps(folder + "/P1.hkl", P.ucell, sg=P.symbol)
    C.miller_array = ma
    C.symbol = "P1" if use_p1 else P.symbol 
    return C

def load_beam(dxtbx_beam, divergence=0):
    """

    :param dxtbx_beam: beam object
    :param divergence: divergence of the beam in degrees
    :return: return a nanoBragg_beam.NBbeam object
    """
    B = nanoBragg_beam.NBbeam()
    B.unit_s0 = dxtbx_beam.get_unit_s0()
    B.flux = 1e12
    B.divergence = divergence
    return B

