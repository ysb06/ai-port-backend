import torch.nn as nn
from efficientnet_pytorch import EfficientNet
from torchvision.models import resnext101_32x8d

class EfficientBase(nn.Module):
    def __init__(
            self, 
            num_classes: int, 
            pretrained_model_name: str = "efficientnet-b7", 
            backbone_freeze=True
        ):
        super(EfficientBase, self).__init__()
        self.backbone = EfficientNet.from_pretrained(pretrained_model_name, num_classes=num_classes)
        self.backbone.requires_grad_(not backbone_freeze)
        self.backbone._fc.requires_grad_(True)    # efficientnet fc
        self.backbone._bn1.requires_grad_(True)
        self.backbone._conv_head.requires_grad_(True)
        self.backbone._blocks[38].requires_grad_(True)

    def forward(self, x):
        return self.backbone(x)


class ResNext(nn.Module):
    def __init__(
            self, 
            num_classes: int, 
            pretrained_model_name: str = "efficientnet-b7", 
            backbone_freeze=True
        ):
        super(ResNext, self).__init__()
        self.backbone = resnext101_32x8d(pretrained=True)
        self.backbone.requires_grad_(not backbone_freeze)
        self.backbone.fc = nn.Linear(in_features=2048, out_features=4096, bias=True)    # resnext fc
        self.classifier = nn.Sequential(
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, num_classes),
        )

    def forward(self, x):
        output = self.backbone(x)
        return self.classifier(output)