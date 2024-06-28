import torch
import torch.nn as nn

class BasicConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding='same', activation_fn=nn.ReLU(), **kwargs):
        super(BasicConv2d, self).__init__()
        if padding == 'same':
            padding = kernel_size // 2
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding, bias=True, **kwargs)
        self.activation_fn = activation_fn

    def forward(self, x):
        x = self.conv(x)
        if self.activation_fn:
            x = self.activation_fn(x)
        return x

class InceptionA(nn.Module):
    def __init__(self, num_channels, c1, c2, c3, c4):
        super(InceptionA, self).__init__()
        self.conv1 = BasicConv2d(num_channels, c1, kernel_size=1)
        self.conv2 = nn.Sequential(
            BasicConv2d(num_channels, c2[0], kernel_size=1),
            BasicConv2d(c2[0], c2[1], kernel_size=5, padding=2)
        )
        self.conv3 = nn.Sequential(
            BasicConv2d(num_channels, c3[0], kernel_size=1),
            BasicConv2d(c3[0], c3[1], kernel_size=3, padding=1),
            BasicConv2d(c3[1], c3[2], kernel_size=3, padding=1)
        )
        self.conv4 = nn.Sequential(
            nn.AvgPool2d(kernel_size=3, stride=1, padding=1),
            BasicConv2d(num_channels, c4, kernel_size=1)
        )

    def forward(self, x):
        return torch.cat([self.conv1(x), self.conv2(x), self.conv3(x), self.conv4(x)], dim=1)

class InceptionB(nn.Module):
    def __init__(self, num_channels, c1, c2):
        super(InceptionB, self).__init__()
        self.conv1 = BasicConv2d(num_channels, c1, kernel_size=3, stride=2, padding=0)
        self.conv2 = nn.Sequential(
            BasicConv2d(num_channels, c2[0], kernel_size=1),
            BasicConv2d(c2[0], c2[1], kernel_size=3, padding=1),
            BasicConv2d(c2[1], c2[2], kernel_size=3, stride=2, padding=0)
        )
        self.conv3 = nn.Sequential(
            nn.MaxPool2d(kernel_size=3, stride=2, padding=0)
        )

    def forward(self, x):
        return torch.cat([self.conv1(x), self.conv2(x), self.conv3(x)], dim=1)

class InceptionC(nn.Module):
    def __init__(self, num_channels, c1, c2, c3, c4):
        super(InceptionC, self).__init__()
        self.conv1 = BasicConv2d(num_channels, c1, kernel_size=1)
        self.conv2 = nn.Sequential(
            BasicConv2d(num_channels, c2[0], kernel_size=1),
            BasicConv2d(c2[0], c2[1], kernel_size=(1, 7), padding=(0, 3)),
            BasicConv2d(c2[1], c2[2], kernel_size=(7, 1), padding=(3, 0))
        )
        self.conv3 = nn.Sequential(
            BasicConv2d(num_channels, c3[0], kernel_size=1),
            BasicConv2d(c3[0], c3[1], kernel_size=(7, 1), padding=(3, 0)),
            BasicConv2d(c3[1], c3[2], kernel_size=(1, 7), padding=(0, 3)),
            BasicConv2d(c3[2], c3[3], kernel_size=(7, 1), padding=(3, 0)),
            BasicConv2d(c3[3], c3[4], kernel_size=(1, 7), padding=(0, 3))
        )
        self.conv4 = nn.Sequential(
            nn.AvgPool2d(kernel_size=3, stride=1, padding=1),
            BasicConv2d(num_channels, c4, kernel_size=1)
        )

    def forward(self, x):
        return torch.cat([self.conv1(x), self.conv2(x), self.conv3(x), self.conv4(x)], dim=1)


