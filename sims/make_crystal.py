
import os
import h5py
from cctbx.array_family import flex
from cctbx import miller, crystal
from simtbx.nanoBragg import nanoBragg_crystal, nanoBragg_beam
from iotbx.reflection_file_reader import any_reflection_file
import numpy as np


from resonet.sims import paths_and_const
from resonet.sims import  process_pdb


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


def get_Nabc(ucell, scale=1):
    """

    :param ucell: unit cell dimension
    :param scale: optionally scale the domain size by a factor (greater than or equal to 1)
    :return: number of unit cells along each unit cell axis, according to domain size of crystal
            see paths_and_const.DOMAINSIZE_MM
    """
    assert scale >= 1
    size_m = scale*paths_and_const.DOMAINSIZE_MM*1e-3
    Na = np.ceil(size_m / (ucell[0]*1e-10))
    Nb = np.ceil(size_m / (ucell[1]*1e-10))
    Nc = np.ceil(size_m / (ucell[2]*1e-10))
    Na = max(Na, 3)
    Nb = max(Nb, 3)
    Nc = max(Nc, 3)
    return Na, Nb, Nc


def load_crystal(folder, rot_mat=None, scale=1):
    """

    :param folder:  pdb folder, e.g. /data/dermen/sims/pdbs/2itu
    :param rot_mat: rotation matrix of crystal (otherwise it will be aligned in the standard PDB convention)
    :param scale: scale factor for mosaic domain size (baseline is DOMAINSIZE_MM in paths_and_const.py)
    :return:
    """
    C = nanoBragg_crystal.NBcrystal(init_defaults=False)
    assert os.path.isdir(folder)
    pdb_base = os.path.basename(folder)
    P = process_pdb.PDB(folder + '/%s.pdb' % pdb_base)
    C.dxtbx_crystal = P.p1_dxtbx_crystal
    C.xtal_shape = "gauss" #"square" #"tophat"
    if rot_mat is not None:
        Umat = C.dxtbx_crystal.get_U()
        Umat = np.dot(rot_mat, np.reshape(Umat,(3,3)))
        C.dxtbx_crystal.set_U(tuple(Umat.ravel()))
    C.Ncells_abc = get_Nabc(P.p1_ucell, scale)
    fmodel_file = os.path.join(folder, "fmodel.mtz")
    #fmodel_file = os.path.join(folder, "fmodel_1p2.mtz")
    ma = any_reflection_file(fmodel_file).as_miller_arrays()[0]
    if ma.is_complex_array():
        ma = ma.as_amplitude_array()
    C.miller_array = ma
    C.symbol = ma.space_group_info().type().lookup_symbol()
    return C


def load_beam(dxtbx_beam, divergence=0):
    """

    :param dxtbx_beam: beam object
    :param divergence: divergence of the beam in degrees
    :return: return a nanoBragg_beam.NBbeam object
    """
    B = nanoBragg_beam.NBbeam()
    B.size_mm = paths_and_const.BEAM_SIZE_MM
    B.unit_s0 = dxtbx_beam.get_unit_s0()
    B.spectrum = [(dxtbx_beam.get_wavelength(), paths_and_const.FLUX)]
    B.flux = paths_and_const.FLUX
    B.divergence = divergence
    return B

