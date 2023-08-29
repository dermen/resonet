

import torch
import torchvision

#initial size: (2527, 2463)
# def tester():
#     print("he")
#
# def mx_alpha():
#     return torch.nn.MaxPool2d(5, stride=1)
#
#
# def mx_beta():
#     return torch.nn.MaxPool2d(3, stride=1)
#
# def nothing():
#     return torch.nn.Identity()
#
# def resize_alpha():
#     '''
#     Forces the dimensions to half the size of the images from
#     '''
#     res = torchvision.transforms.Resize((1263, 1231))
#     return res
#
# def resize_beta():
#     return torchvision.transforms.Resize((842, 821))

def resize_for_trans():
    return torchvision.transforms.Resize((832, 832))

def mx_gamma(dev=None):
    #try ZeroPad2d
    mp = torch.nn.MaxPool2d(3, stride=3)
    if dev is not None:
        mp = mp.to(dev)
    tran = torchvision.transforms.Compose([
        mp,  # outputs image of size (842, 821)
        torchvision.transforms.CenterCrop(832)
    ])
    return tran


