from torchvision.models.segmentation import fcn_resnet50
import torch
from segmentation_models_pytorch import Unet


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


def load_model(model_file):
    info=torch.load(model_file, weights_only=True)
    model_name = info["model_name"]
    if model_name.startswith("eff"):
        model_num = int(model_name.split("-b")[1])
        model = efficientnet(b=model_num)
    elif model_name=="fcn50":
        model = FCN50()
    else:
        raise  NotImplementedError("Do not know how to load model %s" % model_name)
    model.load_state_dict(info["model_state_dict"])
    return model
