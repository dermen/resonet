from resonet import arches
import torch.nn as nn


res18 = lambda *args, **kwargs: arches.RESNetAny(*args, netnum=18, **kwargs)
res34 = lambda *args, **kwargs: arches.RESNetAny(*args, netnum=34, **kwargs)
res50 = lambda *args, **kwargs: arches.RESNetAny(*args, netnum=50, **kwargs)
res101 = lambda *args, **kwargs: arches.RESNetAny(*args, netnum=101, **kwargs)
res152 = lambda *args, **kwargs: arches.RESNetAny(*args, netnum=152, **kwargs)

ARCHES = {"le": arches.LeNet, "res18": res18, "res50": res50,
          "res34": res34, "res101": res101, "res152": res152}

LOSSES = {"L1": nn.L1Loss, "L2": nn.MSELoss, "BCE": nn.BCELoss, "BCE2": nn.BCEWithLogitsLoss}
