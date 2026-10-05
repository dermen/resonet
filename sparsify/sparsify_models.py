from torchvision.models.segmentation import fcn_resnet50
import torch
from segmentation_models_pytorch import Unet
import os

import numpy as np
import numpy._core.multiarray

torch.serialization.add_safe_globals([
    numpy._core.multiarray.scalar,
    np.str_,
    np.dtype,
    np.dtypes.Float64DType,
    np.dtypes.Float32DType,
    np.dtypes.Float16DType,
    np.dtypes.Int64DType,
    np.dtypes.Int32DType,
    np.dtypes.Int16DType,
    np.dtypes.Int8DType,
    np.dtypes.UInt64DType,
    np.dtypes.UInt32DType,
    np.dtypes.UInt16DType,
    np.dtypes.UInt8DType,
    np.dtypes.BoolDType,
    np.dtypes.StrDType,
])

class DownsampleWrapper(torch.nn.Module):
    """Wraps a segmentation model with maxpool downsampling on input
    and nearest-neighbor upsampling on output. This preserves weak signals
    by using maxpool (keeps peak intensities) while reducing spatial dims."""

    def __init__(self, model, factor=2):
        super().__init__()
        self.model = model
        self.factor = factor
        self.pool = torch.nn.MaxPool2d(kernel_size=factor, stride=factor)

    def forward(self, x):
        orig_size = x.shape[2:]  # (H, W)
        x = self.pool(x)
        x = self.model(x)
        x = torch.nn.functional.interpolate(x, size=orig_size, mode='nearest')
        return x


class FCN50(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.model = fcn_resnet50(num_classes=1)
        self.model.backbone.conv1 = \
            torch.nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
        self.sig = torch.nn.Sigmoid()

    def forward(self, x):
        x = self.model(x)['out']
        x = self.sig(x)
        return x

def efficientnet(b=0):
    unet_model = Unet(
        encoder_name="efficientnet-b%d"%b,
        encoder_weights="imagenet",
        in_channels=1, # For grayscale images
        classes=1      # Output 1 channel for binary segmentation
    )
    model = torch.nn.Sequential(unet_model, torch.nn.Sigmoid())
    return model


def default_model_path():
    dirname = os.path.dirname(__file__)
    model_path = os.path.join(dirname, "../downloaded_models/1k_randoms.compress_3_withName.out")
    return model_path


def load_model(model_file=None, map_location=None):
    if model_file is None:
        model_file = default_model_path()
    load_kwargs = {"weights_only": True}
    if map_location is not None:
        load_kwargs["map_location"] = map_location
    info=torch.load(model_file, **load_kwargs)
    model_name = info["model_name"]
    if model_name.startswith("eff"):
        model_num = int(model_name.split("-b")[1])
        model = efficientnet(b=model_num)
    elif model_name=="fcn50":
        model = FCN50()
    else:
        raise  NotImplementedError("Do not know how to load model %s" % model_name)
    downsample_factor = info.get("downsample_factor", None)
    if downsample_factor is not None and downsample_factor > 1:
        model = DownsampleWrapper(model, factor=downsample_factor)
    model.load_state_dict(info["model_state_dict"])
    return model
