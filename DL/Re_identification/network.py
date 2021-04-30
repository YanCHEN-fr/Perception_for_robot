import copy
import torch
import torch.nn as nn
from torchvision.models.resnet import resnet50
import torch.nn.functional as F

num_classes = 751  # change this depend on your dataset


class REID_NET(nn.Module):
    def __init__(self):
        # write the CNN initialization
        super(REID_NET, self).__init__()

        self.fc_hidden1=1024
        self.fc_hidden2=768
        self.resnet = resnet50(pretrained=True)
        self.inchannel = self.resnet.fc.in_features
        self.resnet.fc = nn.Sequential(nn.Linear(self.inchannel, self.fc_hidden1),
                                       nn.ReLU(),
                                       nn.Linear(self.fc_hidden1, self.fc_hidden2))
        self.fc_id = nn.Linear(self.fc_hidden2, num_classes)
        self.fc_metric = nn.Linear(self.fc_hidden2, num_classes)

    def forward(self, x):
        # write the CNN forward
        x = self.resnet(x)  # ResNet

        predict_id= self.fc_id(x) # Write this layer
        predict_metric= self.fc_metric(x) # Write this layer

        predict = torch.cat([predict_id, predict_metric], dim=1)

        return predict, predict_id, predict_metric