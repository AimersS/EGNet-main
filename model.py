import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import math
import numpy as np




class EGNet(nn.Module):
    def __init__(self):
        super(EGNet, self).__init__()

        vgg16ForLeft = torchvision.models.vgg16(pretrained=True)
        vgg16ForRight = torchvision.models.vgg16(pretrained=True)

        self.leftEyeNet = vgg16ForLeft.features
        self.leftPool = nn.AdaptiveAvgPool2d(1)
        self.leftFC = nn.Sequential(
            nn.Linear(512, 1024),
            nn.BatchNorm1d(1024, momentum=0.99, eps=1e-3),
            nn.ReLU(inplace=True)
        )


        self.rightEyeNet = vgg16ForRight.features
        self.rightPool = nn.AdaptiveAvgPool2d(1)
        self.rightFC = nn.Sequential(
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
        leftFeature = self.leftPool(leftFeature)
        leftFeature = leftFeature.view(leftFeature.size(0), -1)
        leftFeature = self.leftFC(leftFeature)

        rightFeature = self.rightEyeNet(x_in['right'])
        rightFeature = self.rightPool(rightFeature)
        rightFeature = rightFeature.view(rightFeature.size(0), -1)  # 全连接前的一个resize 将数据全部输入FC层前做一个整合处理 将特征图变为1张
        rightFeature = self.rightFC(rightFeature)

        feature = torch.cat((leftFeature, rightFeature), 1)      #张量拼接

        feature = self.totalFC1(feature)
        feature = torch.cat((feature, x_in['head_pose']), 1)

        gaze = self.totalFC2(feature)

        return gaze

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_uniform_(m.weight, mode="fan_in", nonlinearity="relu")  #权重初始化
                nn.init.zeros_(m.bias)  #使用常数0对m.bias赋值。


if __name__ == '__main__':
    m = EGNet().cuda()
    #torch.zeros 回一个由标量0填充的张量，它的形状由size决定
    '''feature = {"face":torch.zeros(10, 3, 224, 224).cuda(),  输入图像的size
                "left":torch.zeros(10,1, 36,60).cuda(),
                    "right":torch.zeros(10,1, 36,60).cuda()
              }'''
    feature = {"head_pose": torch.zeros(10, 2).cuda(),
               "left": torch.zeros(10, 3, 36, 60).cuda(),
               "right": torch.zeros(10, 3, 36, 60).cuda()
               }
    a = m(feature)
    print(m)

#study

# studym=cv2.imread(r'D:\Project\Gaze estimation\datatest\Image\p00/face/1.jpg')
# print(m.shape)
# m = self.leftPool(m)

# m= m.view(m.size(0), -1)
# print(m.shape)

# m= self.leftFC(m)
# print(m.shape)

