import torch.nn as nn
from torchvision.models import resnet18, ResNet18_Weights, mobilenet_v2, MobileNet_V2_Weights

def build_model(name, pretrained=True, weights_name=None):
    name = name.lower()
    if name == "resnet18":
        weights = (ResNet18_Weights[weights_name] if weights_name else ResNet18_Weights.DEFAULT) if pretrained else None
        model = resnet18(weights=weights)
        model.fc = nn.Linear(model.fc.in_features, 4)
    elif name == "mobilenetv2":
        weights = (MobileNet_V2_Weights[weights_name] if weights_name else MobileNet_V2_Weights.DEFAULT) if pretrained else None
        model = mobilenet_v2(weights=weights)
        model.classifier[-1] = nn.Linear(model.classifier[-1].in_features, 4)
    else:
        raise ValueError(f"Unsupported model: {name}")
    return model

def freeze_backbone(model, name):
    for p in model.parameters(): p.requires_grad = False
    head = model.fc if name == "resnet18" else model.classifier
    for p in head.parameters(): p.requires_grad = True

def unfreeze_late_layers(model, name):
    freeze_backbone(model, name)
    if name == "resnet18":
        for p in model.layer4.parameters(): p.requires_grad = True
    else:
        for p in model.features[-3:].parameters(): p.requires_grad = True

def frozen_batchnorm_eval(model, name, stage):
    if stage == 1 and name == "resnet18":
        for m in model.modules():
            if isinstance(m, nn.BatchNorm2d): m.eval()
