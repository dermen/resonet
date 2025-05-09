from torchvision.models.segmentation import fcn_resnet50
import torch


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

