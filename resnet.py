import torch
from torchvision.models import resnet18

model = resnet18(pretrained=True)

# Freeze all layers
for param in model.parameters():
    param.requires_grad = False
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
