# """ Implementation of VGG-16 network for deriving content and style loss. """
# from torch import nn, Tensor

# class VGG(nn.Module):

#     def __init__(self, in_channel=3):
#         super(VGG, self).__init__()
#         self.conv1 = self.make_layer(2, in_channel, 64, 3)
#         self.pool1 = nn.AvgPool2d(kernel_size=2, stride=2)
#         self.conv2 = self.make_layer(2, 64, 128, 3)
#         self.pool2 = nn.AvgPool2d(kernel_size=2, stride=2)
#         self.conv3 = self.make_layer(3, 128, 256, 3)
#         self.pool3 = nn.AvgPool2d(kernel_size=2, stride=2)
#         self.conv4 = self.make_layer(3, 256, 512, 3)
#         self.pool4 = nn.AvgPool2d(kernel_size=2, stride=2)
#         self.conv5 = self.make_layer(3, 512, 512, 3)
#         self.pool5 = nn.AvgPool2d(kernel_size=2, stride=2)
#         self.conv6 = Conv2d(512, 4096, kernel_size=7)
#         self.dropout1 = nn.Dropout(0.5)
#         self.conv7 = Conv2d(4096, 4096, kernel_size=1)
#         self.dropout2 = nn.Dropout(0.5)
#         self.conv8 = Conv2d(4096, 1000, kernel_size=1, activation_fn=None)

#     def make_layer(self, repeat, in_channel, out_channel, kernel_size):
#         layer = []
#         for _ in range(repeat):
#             layer.append(Conv2d(in_channel, out_channel, kernel_size=kernel_size))
#             in_channel = out_channel
#         return nn.Sequential(layer)

#     def forward(self, x):
#         """ forward process """
#         x *= 255.0
#         _, _, height, width = x.shape
#         cons = Tensor([123.68, 116.779, 103.939]).unsqeeze(1).unsqeeze(1)
#         cons = cons.repeat(height, 1).repeat(width, 2).unsqeeze(0)
#         x -= cons

#         end_points = {}
#         x = self.conv1(x)
#         end_points['vgg_16/conv1'] = x

#         x = self.pool1(x)
#         x = self.conv2(x)
#         end_points['vgg_16/conv2'] = x

#         x = self.pool2(x)
#         x = self.conv3(x)
#         end_points['vgg_16/conv3'] = x

#         x = self.pool3(x)
#         x = self.conv4(x)
#         end_points['vgg_16/conv4'] = x

#         x = self.pool4(x)
#         x = self.conv5(x)
#         end_points['vgg_16/conv5'] = x

#         x = self.pool5(x)
#         x = self.conv6(x)
#         end_points['vgg_16/conv6'] = x

#         x = self.dropout1(x)
#         x = self.conv7(x)
#         end_points['vgg_16/conv7'] = x

#         x = self.dropout2(x)
#         x = self.conv8(x)
#         end_points['vgg_16/fc8'] = x

#         return end_points

# class Conv2d(nn.Module):
 
#     def __init__(self, in_channels: int, out_channels: int, kernel_size=3, stride=1,
#                  activation_fn=nn.ReLU(), padding_mode='zeros', **kwargs):
#         super(Conv2d, self).__init__()
#         self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride,
#                               padding_mode=padding_mode, **kwargs)
#         self.activation_fn = activation_fn

#     def forward(self, x):
#         x = self.conv(x)
#         if self.activation_fn:
#             x = self.activation_fn(x)
#         return x

import torch
import torch.nn as nn
from torchvision import models
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

# class VGGencoder(nn.Module):
#     def __init__(self, pretrained=True):
#         super(VGGencoder, self).__init__()
#         vgg16 = models.vgg16(pretrained=pretrained)
#         self.features = vgg16.features
#         self.classifier = nn.Sequential(
#             nn.Linear(512 * 7 * 7, 4096),
#             nn.ReLU(True),
#             nn.Dropout(),
#             nn.Linear(4096, 4096),
#             nn.ReLU(True),
#             nn.Dropout(),
#             nn.Linear(4096, 1000),
#         )
#         self.avgpool = vgg16.avgpool
    
#     def forward(self, x):
#         x = self.features(x)
#         x = self.avgpool(x)
#         print("vgg에서: ", x.shape)
#         x = torch.flatten(x)
#         print("flatten하면 ", x.shape)
#         x = self.classifier(x)
#         return x

#     def extract_features(self, x):
#         end_points = {}
#         x = self.features[0:2](x)
#         end_points['vgg_16/conv1'] = x
#         x = self.features[2:5](x)
#         end_points['vgg_16/conv2'] = x
#         x = self.features[5:10](x)
#         end_points['vgg_16/conv3'] = x
#         x = self.features[10:15](x)
#         end_points['vgg_16/conv4'] = x
#         x = self.features[15:20](x)
#         end_points['vgg_16/conv5'] = x
#         x = self.avgpool(x)
#         x = torch.flatten(x, 1)
#         x = self.classifier[0:2](x)
#         end_points['vgg_16/conv6'] = x
#         x = self.classifier[2:5](x)
#         end_points['vgg_16/conv7'] = x
#         x = self.classifier[5:](x)
#         end_points['vgg_16/conv8'] = x
#         return end_points

class VGGencoder(nn.Module):
    def __init__(self, in_channel=3):
        super(VGGencoder, self).__init__()
        self.conv1 = self.make_layer(2, in_channel, 64, 3)
        self.pool1 = nn.AvgPool2d(kernel_size=2, stride=2)
        self.conv2 = self.make_layer(2, 64, 128, 3)
        self.pool2 = nn.AvgPool2d(kernel_size=2, stride=2)
        self.conv3 = self.make_layer(3, 128, 256, 3)
        self.pool3 = nn.AvgPool2d(kernel_size=2, stride=2)
        self.conv4 = self.make_layer(3, 256, 512, 3)
        self.pool4 = nn.AvgPool2d(kernel_size=2, stride=2)
        self.conv5 = self.make_layer(3, 512, 512, 3)
        self.pool5 = nn.AvgPool2d(kernel_size=2, stride=2)
        self.conv6 = Conv2d(512, 4096, kernel_size=7)
        self.dropout1 = nn.Dropout(0.5)
        self.conv7 = Conv2d(4096, 4096, kernel_size=1)
        self.dropout2 = nn.Dropout(0.5)
        self.conv8 = Conv2d(4096, 1000, kernel_size=1, activation_fn=None) # 여기도 None에서 ReLU로 통일해주었음
    
    def make_layer(self, repeat, in_channel, out_channel, kernel_size):
        layer = []
        for _ in range(repeat):
            layer.append(Conv2d(in_channel, out_channel, kernel_size=kernel_size))
            
            in_channel = out_channel
        return nn.Sequential(*layer)

    def forward(self, x):
        x = x.to(device)
        x *= 255.0
        # print("vgg 인풋 쉐입: ", x.shape)
        
        _, _, height, width = x.shape
        cons = torch.Tensor([123.68, 116.779, 103.939]).unsqueeze(1).unsqueeze(1)
        cons = cons.to(device)
        # print("unsqueeze하면 ", cons.shape)
        cons = cons.repeat(1, height, width)
        # print("뭐 이상한 전처리하면 ", cons.shape)
        x -= cons
        end_points = {}
        x = self.conv1(x)
        end_points['vgg_16/conv1'] = x
        x = self.pool1(x)
        x = self.conv2(x)
        end_points['vgg_16/conv2'] = x
        x = self.pool2(x)
        x = self.conv3(x)
        end_points['vgg_16/conv3'] = x
        x = self.pool3(x)
        x = self.conv4(x)
        end_points['vgg_16/conv4'] = x
        x = self.pool4(x)
        x = self.conv5(x)
        end_points['vgg_16/conv5'] = x
        x = self.pool5(x)
        x = self.conv6(x)
        end_points['vgg_16/conv6'] = x
        x = self.dropout1(x)
        x = self.conv7(x)
        end_points['vgg_16/conv7'] = x
        x = self.dropout2(x)
        x = self.conv8(x)
        end_points['vgg_16/fc8'] = x
        # print("VGG에서", end_points.shape)
        return end_points
    
class Conv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, 
                 activation_fn=nn.ReLU(), padding='same', **kwargs):
        super(Conv2d, self).__init__()
        if padding == 'same':
            padding= kernel_size//2
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride,
                              padding=padding, bias=True, **kwargs)
        self.activation_fn = activation_fn
    def forward(self,x):
        x=self.conv(x)
        if self.activation_fn:
            x = self.activation_fn(x)
        return x 

# class VGGencoder(nn.Module):
#     def __init__(self, in_channel=3):
#         super(VGGencoder, self).__init__()
#         vgg = models.vgg16(pretrained=True)
#         self.features = nn.Sequential(*list(vgg.features.children())[:9])  # 23번째 레이어까지 사용
#         self.conv1 = self.make_layer(2, in_channel, 64, 3)
#         self.pool1 = nn.AvgPool2d(kernel_size=2, stride=2)
#         self.conv2 = self.make_layer(2, 64, 128, 3)
#         self.pool2 = nn.AvgPool2d(kernel_size=2, stride=2)
#         self.conv3 = self.make_layer(3, 128, 256, 3)
#         self.pool3 = nn.AvgPool2d(kernel_size=2, stride=2)
#         self.conv4 = self.make_layer(3, 256, 512, 3)
#         self.pool4 = nn.AvgPool2d(kernel_size=2, stride=2)
#         self.conv5 = self.make_layer(3, 512, 512, 3)
#         self.pool5 = nn.AvgPool2d(kernel_size=2, stride=2)
#         self.conv6 = nn.Conv2d(512, 4096, kernel_size=7, padding=7//2)
#         self.dropout1 = nn.Dropout(0.5)
#         self.conv7 = nn.Conv2d(4096, 4096, kernel_size=1, padding=1//2)
#         self.dropout2 = nn.Dropout(0.5)
#         self.conv8 = nn.Conv2d(4096, 1000, kernel_size=1, activation_fn=None, padding=1//2) # 여기도 None에서 ReLU로 통일해주었음
    
#     def make_layer(self, repeat, in_channel, out_channel, kernel_size):
#         layer = []
#         for _ in range(repeat):
#             layer.append(nn.Conv2d(in_channel, out_channel, kernel_size=kernel_size, padding=kernel_size//2))
            
#             in_channel = out_channel
#         return nn.Sequential(*layer)
    
#     def forward(self, x):
#         x = x.to(device)
#         x *= 255.0
#         # print("vgg 인풋 쉐입: ", x.shape)
        
#         _, _, height, width = x.shape
#         cons = torch.Tensor([123.68, 116.779, 103.939]).unsqueeze(1).unsqueeze(1)
#         cons = cons.to(device)
#         # print("unsqueeze하면 ", cons.shape)
#         cons = cons.repeat(1, height, width)
#         # print("뭐 이상한 전처리하면 ", cons.shape)
#         x -= cons
#         end_points = {}
#         x = self.conv1(x)
#         end_points['vgg_16/conv1'] = x
#         x = self.pool1(x)
#         x = self.conv2(x)
#         end_points['vgg_16/conv2'] = x
#         x = self.pool2(x)
#         x = self.conv3(x)
#         end_points['vgg_16/conv3'] = x
#         x = self.pool3(x)
#         x = self.conv4(x)
#         end_points['vgg_16/conv4'] = x
#         x = self.pool4(x)
#         x = self.conv5(x)
#         end_points['vgg_16/conv5'] = x
#         x = self.pool5(x)
#         x = self.conv6(x)
#         end_points['vgg_16/conv6'] = x
#         x = self.dropout1(x)
#         x = self.conv7(x)
#         end_points['vgg_16/conv7'] = x
#         x = self.dropout2(x)
#         x = self.conv8(x)
#         end_points['vgg_16/fc8'] = x
#         print("VGG에서", end_points.shape)
#         return end_points