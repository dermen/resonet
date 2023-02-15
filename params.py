from resonet import arches
import torch.nn as nn

ARCHES = {"le": arches.LeNet, "res18": arches.RESNet18, "res50": arches.RESNet50, "res50bc": arches.RESNet50BC,
          "res34": arches.RESNet34, "res101": arches.RESNet101, "res152": arches.RESNet152}
LOSSES = {"L1": nn.L1Loss, "L2": nn.MSELoss, "BCE": nn.BCELoss, "BCE2": nn.BCEWithLogitsLoss}

