

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
    from sixdrepnet import utils
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
    from sixdrepnet.loss import GeodesicLoss
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


if __name__ == "__main__":
    test_loss()
    test_sixdrep()
    test_batch_cross()
    test_rotmat_distance()