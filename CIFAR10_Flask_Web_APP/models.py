import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from torchvision import models
import numpy as np
import matplotlib.pyplot as plt
import os
import random
import matplotlib.gridspec as gridspec
from matplotlib.ticker import MaxNLocator
import PIL
import sys
import seaborn as sns
import sklearn.metrics
import pickle
import warnings
warnings.filterwarnings("ignore")


def set_parameter_requires_grad(model, feature_extracting):
    if feature_extracting:
        for param in model.parameters():
            param.requires_grad = False

class Interpolate(nn.Module):
    def __init__(self, size, mode):
        super(Interpolate, self).__init__()
        self.interp = nn.functional.interpolate
        self.size = size
        self.mode = mode
        
    def forward(self, x):
        x = self.interp(x, size=self.size, mode=self.mode, align_corners=False)
        return x

class GoogLeNetModel(nn.Module):
    def __init__(self, n_class, fine_tune, pretrained=True):
        super(GoogLeNetModel, self).__init__()
        
        self.input = Interpolate((224,224),'bilinear')
        
        #load the pretrained model
        googleNet = models.googlenet(pretrained=pretrained)
        # delete the last fc layer.
        modules = list(googleNet.children())[:-1]      
        self.model = nn.Sequential() 
        self.model.features = nn.Sequential(*modules)
        set_parameter_requires_grad(self.model, fine_tune)

        #adding 2 linear layers
        self.model.classifier = nn.Sequential(
                    nn.Dropout(),
                    nn.Linear(in_features=1024, out_features=512, bias=True),
                    nn.Dropout(),
                    nn.ReLU(),
                    nn.Linear(in_features=512, out_features=n_class, bias=True)
                        )

    def forward(self, x):
      
        x = self.input(x)
        x_ch0 = torch.unsqueeze(x[:, 0], 1) * (0.229 / 0.5) + (0.485 - 0.5) / 0.5
        x_ch1 = torch.unsqueeze(x[:, 1], 1) * (0.224 / 0.5) + (0.456 - 0.5) / 0.5
        x_ch2 = torch.unsqueeze(x[:, 2], 1) * (0.225 / 0.5) + (0.406 - 0.5) / 0.5
        x = torch.cat((x_ch0, x_ch1, x_ch2), 1)
        
        x = self.model.features(x)
        
        x = x.view(x.size(0), -1)
        out = self.model.classifier(x)

        return out

class AlexModel(nn.Module):
    def __init__(self, n_class, fine_tune, pretrained=True):
        super(AlexModel, self).__init__()
        
        self.input = Interpolate((227,227),'bilinear')
        #load the pretrained model
        alexNet = models.alexnet(pretrained=pretrained)
        self.model = nn.Sequential()
        self.model.features = alexNet.features
        self.avgpool = alexNet.avgpool
        set_parameter_requires_grad(self.model, fine_tune)

        #adding 2 linear layers
        self.model.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(256 * 6 * 6, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, n_class))
            
            
    def forward(self, x):
        x = self.input(x)
        x = self.model.features(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), 256 * 6 * 6)
        out = self.model.classifier(x)

        return out

class ResNetModel(nn.Module):
    def __init__(self, n_class, fine_tune, pretrained=True):
        super(ResNetModel, self).__init__()
        
        self.input = Interpolate((224,224),'bilinear')
        
        #load the pretrained model
        resnet = models.resnet34(pretrained=pretrained)
        # delete the last fc layer.
        modules = list(resnet.children())[:-1]      
        self.model = nn.Sequential() 
        self.model.features = nn.Sequential(*modules)
        set_parameter_requires_grad(self.model, fine_tune)

        #adding 2 linear layers
        self.model.classifier = nn.Sequential(
                    nn.Dropout(),
                    nn.Linear(512*1, 512, bias=True),
                    nn.Dropout(),
                    nn.ReLU(),
                    nn.Linear(in_features= 512, out_features=n_class, bias=True)
                        )

    def forward(self, x):
      
        x = self.input(x)
        x = self.model.features(x)
        x = x.view(x.size(0), -1)
        out = self.model.classifier(x)

        return out

class InceptionModel(nn.Module):
    def __init__(self, n_class, fine_tune, pretrained=True):
        super(InceptionModel, self).__init__()
        
        self.input = Interpolate((299,299),'bilinear')
        
        #load the pretrained model
        inceptionNet = models.inception_v3(pretrained=pretrained)
        # delete the last fc layer.
        modules = list(inceptionNet.children())[:-1]      
        self.model = nn.Sequential() 
        self.model.features = nn.Sequential(*modules)
        set_parameter_requires_grad(self.model, fine_tune)

        #adding 2 linear layers
        self.model.classifier = nn.Sequential(
                    nn.Dropout(),
                    nn.Linear(in_features=2048, out_features=512, bias=True),
                    nn.Dropout(),
                    nn.ReLU(),
                    nn.Linear(in_features=512, out_features=n_class, bias=True)
                        )

    def forward(self, x):
      
        x = self.input(x)
        x_ch0 = torch.unsqueeze(x[:, 0], 1) * (0.229 / 0.5) + (0.485 - 0.5) / 0.5
        x_ch1 = torch.unsqueeze(x[:, 1], 1) * (0.224 / 0.5) + (0.456 - 0.5) / 0.5
        x_ch2 = torch.unsqueeze(x[:, 2], 1) * (0.225 / 0.5) + (0.406 - 0.5) / 0.5
        x = torch.cat((x_ch0, x_ch1, x_ch2), 1)
        
        x = self.model.features(x)
        
        x = x.view(x.size(0), -1)
        out = self.model.classifier(x)

        return out

class VggModel(nn.Module):
    def __init__(self, n_class, fine_tune, pretrained=True):
        super(VggModel, self).__init__()
        
        #load the pretrained model
        vgg11_bn = models.vgg19_bn(pretrained=pretrained)
        self.model = nn.Sequential()
        self.model.features = vgg11_bn.features
        set_parameter_requires_grad(self.model, fine_tune)

        #adding 2 linear layers
        self.model.classifier = nn.Sequential(
                    nn.Linear(in_features=512, out_features=256, bias=True),
                    nn.BatchNorm1d(256),
                    nn.ReLU(),
                    nn.Linear(in_features=256, out_features=n_class, bias=True)
                        )

    def forward(self, x):

        x = self.model.features(x)
        x = x.squeeze()
        out = self.model.classifier(x)

        return out

