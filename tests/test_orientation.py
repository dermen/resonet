

import pytest
import torch
import numpy as np
from scipy.spatial.transform import Rotation

from resonet.utils import orientation


def test_batch_cross():
    Nbatch=5096
    rot_mats = Rotation.random(num=Nbatch, random_state=0).as_matrix()
    col1 = torch.tensor(rot_mats[:, :, 0])
    col2 = torch.tensor(rot_mats[:, :, 1])
    col3 = orientation.batch_cross(col1, col2)
    assert np.allclose((col1*col3).sum(1).numpy(), 0)
    assert np.allclose((col2*col3).sum(1).numpy(), 0)
    print("ok")


def test_rotmat_distance():

    Nbatch=5096
    rot_mats = Rotation.random(num=Nbatch, random_state=0).as_matrix()
    col1 = rot_mats[:, :, 0]
    col2 = rot_mats[:, :, 1]
    model_outputs = torch.tensor(np.hstack((col1, col2)))
    new_rots = orientation.gs_mapping(model_outputs)
    loss = orientation.loss(new_rots, new_rots).item() / Nbatch
    print("Loss: ", loss)
    assert loss < 1e-3
    print("ok")


def test_sixdrep():
    try:
        from sixdrepnet import utils
    except:
        pytest.skip("sixdrepnet is not installed")
    Nbatch = 5096
    rot_mats = Rotation.random(num=Nbatch, random_state=0).as_matrix()
    col1 = rot_mats[:, :, 0]
    col2 = rot_mats[:, :, 1]
    poses = torch.tensor(np.hstack((col1, col2)))
    new_rots = orientation.gs_mapping(poses)
    test = utils.compute_rotation_matrix_from_ortho6d(poses)
    assert torch.allclose(test, new_rots.view((-1, 3, 3)))
    print("ok")


def test_loss():
    try:
        from sixdrepnet.loss import GeodesicLoss
    except:
        pytest.skip("sixdrepnet is not installed")
    test_loss = GeodesicLoss()
    Nbatch = 5096
    rot_mats = Rotation.random(num=Nbatch, random_state=0).as_matrix()
    col1 = rot_mats[:, :, 0]
    col2 = rot_mats[:, :, 1]
    poses = torch.tensor(np.hstack((col1, col2)))
    rots = orientation.gs_mapping(poses)
    rots_T = rots.transpose(1,2)
    test_l = test_loss(rots, rots_T)

    l = orientation.loss(rots, rots_T)
    print(test_l.item(), l.item())
    assert np.allclose(test_l.item(), l.item())
    print("loss ok")


def test_angles():
    #from sixdrepnet.loss import GeodesicLoss
    #test_loss = GeodesicLoss()
    Nbatch = 5096
    rot_mats = Rotation.random(num=Nbatch, random_state=0).as_matrix()
    degs = np.arange(0, 181, 2)
    for deg in degs:
        # choose N perterbation matrices
        gvecs = np.random.normal(0, 1, (len(rot_mats), 3))
        unit_vecs = gvecs / np.linalg.norm(gvecs, axis=1)[:,None]
        rot_vecs = unit_vecs * deg
        offsets = Rotation.from_rotvec(rot_vecs, degrees=True).as_matrix()
        rot_mats2 = np.array([np.dot(R2, R) for R2, R in zip(offsets, rot_mats)])
        l = orientation.loss(torch.tensor(rot_mats), torch.tensor(rot_mats2))
        ang = l.item()*180 / np.pi
        assert round(ang) == deg
        #l2 = test_loss(torch.tensor(rot_mats), torch.tensor(rot_mats2))
        #ang2 = l2.item()*180 / np.pi
        #assert round(ang2) == deg

    print("ok")


def test_dxtbx():
    # dxtbx crystal description
    try:
        from dxtbx.model.crystal import CrystalFactory
    except:
        pytest.skip("Dxtbx is not installed")
    np.random.seed(0)

    a = np.array([79,0,0])
    b = np.array([0,79,0])
    c = np.array([0,0,38])

    # randomly orient the crystal
    gvec = np.random.normal(0, 1, 3)
    unit_vec = gvec / np.linalg.norm(gvec)
    rot_vec = unit_vec * np.random.uniform(0,180)
    init_rotmat = Rotation.from_rotvec(rot_vec, degrees=True).as_matrix()

    a2 = np.dot(init_rotmat, a)
    b2 = np.dot(init_rotmat, b)
    c2 = np.dot(init_rotmat, c)
    cryst_descr = {'__id__': 'crystal',
                   'real_space_a': tuple(a2),
                   'real_space_b': tuple(b2),
                   'real_space_c': tuple(c2),
                   'space_group_hall_symbol': '-P 4 2'}

    init_C = CrystalFactory.from_dict(cryst_descr)
    init_U = np.reshape(init_C.get_U(), (3,3))[None]

    # now perturb it several times
    for i in range(100):
        gvec = np.random.normal(0, 1, 3)
        unit_vec = gvec / np.linalg.norm(gvec)
        deg = np.random.uniform(0, 180)
        rot_vec = unit_vec * deg
        rotmat = Rotation.from_rotvec(rot_vec, degrees=True).as_matrix()

        a3 = np.dot(rotmat, a2)
        b3 = np.dot(rotmat, b2)
        c3 = np.dot(rotmat, c2)
        cryst_descr = {'__id__': 'crystal',
                       'real_space_a': tuple(a3),
                       'real_space_b': tuple(b3),
                       'real_space_c': tuple(c3),
                       'space_group_hall_symbol': '-P 4 2'}

        C = CrystalFactory.from_dict(cryst_descr)
        U = np.reshape(C.get_U(), (3,3))[None]
        l = orientation.loss( torch.tensor(U), torch.tensor(init_U) )
        assert np.allclose(deg, l.item()*180/np.pi)

    print("dxtbx ok")

if __name__ == "__main__":
    test_angles()
    test_loss()
    test_sixdrep()
    test_batch_cross()
    test_rotmat_distance()
    test_dxtbx()
