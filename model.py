"""Model definitions and checkpoint helpers.

Two kinds of model are supported:
  * ``convnet``  - a small CNN trained from scratch (the original project model,
                   now with BatchNorm and global average pooling).
  * pretrained   - an ImageNet backbone (EfficientNet / ResNet) with a new
                   classification head, fine-tuned on the mushroom data.
"""
import torch
import torch.nn as nn
import torchvision.models as tvm

ARCHS = ("efficientnet_b0", "resnet18", "resnet50", "convnet")

# name of the attribute that holds each architecture's classification head
_HEAD_ATTR = {
    "efficientnet_b0": "classifier",
    "resnet18": "fc",
    "resnet50": "fc",
}


def _conv_block(c_in, c_out):
    # conv -> batchnorm -> relu -> halve the spatial size
    return nn.Sequential(
        nn.Conv2d(c_in, c_out, kernel_size=3, padding=1, bias=False),
        nn.BatchNorm2d(c_out),
        nn.ReLU(inplace=True),
        nn.MaxPool2d(2),
    )


class ConvNet(nn.Module):
    """Small from-scratch CNN.

    Global average pooling replaces the old 256*14*14 -> 512 linear layer,
    which alone held ~25.7M parameters and was the main source of overfitting.
    """

    def __init__(self, num_classes, dropout=0.3):
        super().__init__()
        self.features = nn.Sequential(
            _conv_block(3, 32),
            _conv_block(32, 64),
            _conv_block(64, 128),
            _conv_block(128, 256),
            _conv_block(256, 512),
        )
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(dropout),
            nn.Linear(512, num_classes),
        )

    def forward(self, x):
        return self.classifier(self.pool(self.features(x)))


def build_model(arch, num_classes, pretrained=True):
    """Create a model with a head sized for ``num_classes``."""
    if arch == "convnet":
        return ConvNet(num_classes)

    if arch == "efficientnet_b0":
        weights = tvm.EfficientNet_B0_Weights.DEFAULT if pretrained else None
        model = tvm.efficientnet_b0(weights=weights)
        in_features = model.classifier[1].in_features
        model.classifier = nn.Sequential(nn.Dropout(0.3), nn.Linear(in_features, num_classes))
        return model

    if arch in ("resnet18", "resnet50"):
        if arch == "resnet18":
            weights = tvm.ResNet18_Weights.DEFAULT if pretrained else None
            model = tvm.resnet18(weights=weights)
        else:
            weights = tvm.ResNet50_Weights.DEFAULT if pretrained else None
            model = tvm.resnet50(weights=weights)
        in_features = model.fc.in_features
        model.fc = nn.Sequential(nn.Dropout(0.3), nn.Linear(in_features, num_classes))
        return model

    raise ValueError(f"Unknown arch {arch!r}; choose from {ARCHS}")


def head_parameters(model, arch):
    """Parameters of the classification head (everything, for convnet)."""
    if arch not in _HEAD_ATTR:
        return list(model.parameters())
    return list(getattr(model, _HEAD_ATTR[arch]).parameters())


def set_backbone_trainable(model, arch, trainable):
    """Freeze or unfreeze everything except the classification head."""
    if arch not in _HEAD_ATTR:
        return  # from-scratch model: always fully trainable
    head_ids = {id(p) for p in head_parameters(model, arch)}
    for p in model.parameters():
        if id(p) not in head_ids:
            p.requires_grad = trainable


def save_checkpoint(path, model, arch, classes, extra=None):
    """Save weights together with everything needed to rebuild the model."""
    torch.save(
        {
            "arch": arch,
            "classes": list(classes),
            "state_dict": model.state_dict(),
            **(extra or {}),
        },
        path,
    )


def load_checkpoint(path, device="cpu"):
    """Rebuild a model from a checkpoint. Returns (model, classes, checkpoint)."""
    ckpt = torch.load(path, map_location=device, weights_only=True)
    model = build_model(ckpt["arch"], len(ckpt["classes"]), pretrained=False)
    model.load_state_dict(ckpt["state_dict"])
    model.to(device).eval()
    return model, ckpt["classes"], ckpt
