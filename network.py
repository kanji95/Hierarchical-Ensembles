import torch
import torch.nn as nn

import torchvision.models as models

class ResNet18(nn.Module):
    """Used for Cross-entropy, CRM, Making-better-mistakes."""

    def __init__(self, model, feature_size, num_classes):
        super(ResNet18, self).__init__()

        self.features_2 = nn.Sequential(*list(model.children())[:-2])
        self.max = nn.MaxPool2d(kernel_size=7, stride=7)
        self.num_ftrs = 512
        
        self.features_1 = nn.Sequential(
            nn.BatchNorm1d(self.num_ftrs),
            nn.Linear(self.num_ftrs, feature_size),
            nn.BatchNorm1d(feature_size),
            nn.ELU(inplace=True),
        )
        self.classifier_3 = nn.Sequential(
            nn.Linear(feature_size, num_classes), )

    def forward(self, x):
        x = self.features_2(x)
        x = self.max(x)
        x = x.view(x.size(0), -1)
        x = self.features_1(x)
        y = self.classifier_3(x)
        return y
    
def load_checkpoint(arch, num_classes, checkpoint_path):
    model = models.__dict__[arch](pretrained=True)
    model = ResNet18(model, feature_size=600, num_classes=num_classes)

    print(f'Using checkpoint at {checkpoint_path}!')
    checkpoint = torch.load(checkpoint_path, map_location=torch.device('cuda'))
    checkpoint = checkpoint["state_dict"]

    model = nn.DataParallel(model)
    model.load_state_dict(checkpoint)
    model = model.to('cuda')
    model.eval();
    
    return model