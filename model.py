import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import math
import numpy as np
#from Torch.ntools import VGG16

class model(nn.Module):
    def __init__(self):
        super(model, self).__init__()
        # 36 * 60
        # 18 * 30
        # 9  * 15
        
        vgg16ForLeft = torchvision.models.vgg16(pretrained=True)
        vgg16ForRight = torchvision.models.vgg16(pretrained=True)
        vgg16ForFace = torchvision.models.vgg16(pretrained=True)

        self.leftEyeNet = vgg16ForLeft.features
        self.rightEyeNet = vgg16ForRight.features
        self.faceNet = vgg16ForFace.features

        self.leftPool = nn.AdaptiveAvgPool2d(1)
        self.rightPool = nn.AdaptiveAvgPool2d(1)
        self.facePool = nn.AdaptiveAvgPool2d(1)

        self.leftFC = nn.Sequential(
            nn.Linear(512, 1024),
            nn.BatchNorm1d(1024, momentum=0.99, eps=1e-3),
            nn.ReLU(inplace=True)
        )
        self.rightFC = nn.Sequential(
            nn.Linear(512, 1024),
            nn.BatchNorm1d(1024, momentum=0.99, eps=1e-3),
            nn.ReLU(inplace=True)
        )

        self.faceFC = nn.sequential(
            nn.Linear(512, 1024),
            nn.BatchNorm1d(1024, momentum=0.99, eps=1e-3),
            nn.ReLU(inplace=True)
         )

        self.totalFC1 = nn.Sequential(
            nn.Linear(2048, 512),
            nn.BatchNorm1d(512, momentum=0.99, eps=1e-3),
            nn.ReLU(inplace=True)
        )

        self.totalFC2 = nn.Sequential(
            nn.Linear(514, 256),
            nn.BatchNorm1d(256, momentum=0.99, eps=1e-3),
            nn.ReLU(inplace=True),
            nn.Linear(256, 2)
        )
        self._init_weights()

    def forward(self, x_in):
        leftFeature = self.leftEyeNet(x_in['left'])
        print('left',leftFeature.shape)
        leftFeature = self.leftPool(leftFeature)
        print('leftPool',leftFeature.shape)
        leftFeature = leftFeature.view(leftFeature.size(0), -1)
        print('leftview',leftFeature.shape)
        leftFeature = self.leftFC(leftFeature)

        rightFeature = self.rightEyeNet(x_in['right'])
        rightFeature = self.rightPool(rightFeature)
        rightFeature = rightFeature.view(rightFeature.size(0), -1)
        rightFeature = self.rightFC(rightFeature)

        faceFeature = self.faceNet(x_in['face'])
        print('faec',faceFeature.shape)
        faceFeature = self.facePool(faceFeature)
        faceFeature = faceFeature.view(faceFeature.size(0),-1)
        print('view',faceFeature.shape)
        faceFeature = self.faceFC(faceFeature)


        feature = torch.cat((leftFeature, rightFeature), 1)

        feature = self.totalFC1(feature)
        feature = torch.cat((feature, faceFeature), 1)

        gaze = self.totalFC2(feature)

        return gaze

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_uniform_(m.weight, mode="fan_in", nonlinearity="relu")
                nn.init.zeros_(m.bias)

if __name__ == '__main__':
    m = model().cuda()
    '''feature = {"face":torch.zeros(10, 3, 224, 224).cuda(),
                "left":torch.zeros(10,1, 36,60).cuda(),
                "right":torch.zeros(10,1, 36,60).cuda()
              }'''
    feature = {"face": torch.zeros(10, 3, 224, 224).cuda(),
               "left": torch.zeros(10, 3, 36, 60).cuda(),
               "right": torch.zeros(10, 3, 36, 60).cuda()
               }
    a = m(feature)
    print(m)
