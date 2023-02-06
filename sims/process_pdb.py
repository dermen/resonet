
import numpy as np

from iotbx import pdb
from dxtbx.model import CrystalFactory


class PDB:
    """lightweight class to help parsing PDBs with CCTBX"""
    def __init__(self, name):
        """

        :param name:  path to pdb file
        """
        self.name = name
        self.P = pdb.input(name)
        
    @property
    def sym(self):
        """
        symmetry of the lattice
        :return:
        """
        return self.P.crystal_symmetry()

    @property
    def hall(self):
        """
        hall symbol of the lattice
        """
        return self.sym.space_group_info().type().hall_symbol()

    @property
    def symbol(self):
        """lookup symbol"""
        return self.sym.space_group_info().type().lookup_symbol()

    @property
    def dxtbx_crystal(self):
        """crystal model in the high symmetry"""
        a,b,c = self.real_space_vectors
        cryst_descr = {'__id__': 'crystal',
                      'real_space_a': a,
                      'real_space_b': b,
                      'real_space_c': c,
                      'space_group_hall_symbol': self.hall}
        return CrystalFactory.from_dict(cryst_descr)

    @property
    def p1_dxtbx_crystal(self):
        """crystal model in the P1 space group"""
        p1_C = self.dxtbx_crystal.change_basis(self.to_p1_op)
        return p1_C

    @property
    def ucell(self):
        """ucell dimension in the high symmetry (a,b,c,alpha,beta,gamme) Angstroms/degrees"""
        return self.sym.unit_cell().parameters()

    @property
    def to_p1_op(self):
        """cctbx operator to convert to P1"""
        return self.sym.change_of_basis_op_to_primitive_setting()

    @property
    def p1_ucell(self):
        """ucell dimension in P1 space group (a,b,c,alpha,beta,gamme) Angstroms/degrees"""
        return self.sym.change_basis(self.to_p1_op).unit_cell().parameters()

    @property
    def real_space_vectors(self):
        """crystal lattice vectors a,b,c"""
        return map(tuple, np.reshape(self.sym.unit_cell().orthogonalization_matrix(),(3,3)).T)

