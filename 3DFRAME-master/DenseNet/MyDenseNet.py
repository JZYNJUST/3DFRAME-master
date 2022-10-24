import torch
import torch.nn as nn
import torch.nn.functional as F
import random
from collections import OrderedDict


class _DenseLayer(nn.Sequential):
    def __init__(self, in_channels, growth_rate, bn_size):
        super(_DenseLayer, self).__init__()
        # self.add_module('norm1', nn.BatchNorm2d(in_channels))
        self.add_module('conv1', nn.Conv2d(in_channels, bn_size * growth_rate,
                                           kernel_size=1,
                                           stride=1, bias=False))
        self.add_module('relu1', nn.LeakyReLU(0.2,inplace=True))

        # self.add_module('norm2', nn.BatchNorm2d(bn_size * growth_rate))
        self.add_module('conv2', nn.Conv2d(bn_size * growth_rate, growth_rate,
                                           kernel_size=3,
                                           stride=1, padding=1, bias=False))
        self.add_module('relu2', nn.LeakyReLU(0.2,inplace=True))


    # 重载forward函数
    def forward(self, x):
        new_features = super(_DenseLayer, self).forward(x)
        return torch.cat([x, new_features], 1)


class _DenseBlock(nn.Sequential):
    def __init__(self, num_layers, in_channels, bn_size, growth_rate):
        super(_DenseBlock, self).__init__()
        for i in range(num_layers):
            self.add_module('denselayer%d' % (i + 1),
                            _DenseLayer(in_channels + growth_rate * i,
                                        growth_rate, bn_size))


class _Transition(nn.Sequential):
    def __init__(self, in_channels, out_channels):
        super(_Transition, self).__init__()
        self.add_module('norm', nn.BatchNorm2d(in_channels))
        self.add_module('relu', nn.ReLU(inplace=True))
        self.add_module('conv', nn.Conv2d(in_channels, out_channels,
                                          kernel_size=1,
                                          stride=1, bias=False))
        self.add_module('pool', nn.MaxPool2d(kernel_size=2, stride=2))


class MyDenseNet(nn.Module):
    def __init__(self, growth_rate=2, block_config=(6, 12, 24, 48, 96, 128, 256, 512), bn_size=1, theta=0.5, num_points=84):
        super(MyDenseNet, self).__init__()

        self.use_cuda = True
        self.random = random.random

        num_init_feature = 2 * growth_rate
        self.features = nn.Sequential(OrderedDict([
            ("conv0", nn.Conv2d(in_channels=1, out_channels=num_init_feature, kernel_size=7, padding=3)),
            ("bn0", nn.BatchNorm2d(num_init_feature)),
            ("relu0", nn.ReLU(inplace=True))
        ]))

        num_feature = num_init_feature
        for i, num_layers in enumerate(block_config):
            self.features.add_module('denseblock%d' % (i + 1),
                                     _DenseBlock(num_layers, num_feature,
                                                 bn_size, growth_rate))
            # 计算当前特征图通道数目
            num_feature = num_feature + growth_rate * num_layers
            if i != len(block_config) - 1:
                self.features.add_module('transition%d' % (i + 1),
                                         _Transition(num_feature,
                                                     int(num_feature * theta)))
                num_feature = int(num_feature * theta)
        # print("num_features = ",num_feature)
        self.features.add_module('norm4', nn.BatchNorm2d(num_feature))
        self.features.add_module('relu4', nn.ReLU(inplace=True))
        self.features.add_module('avg_pool', nn.MaxPool2d(kernel_size=2, stride=2))
        self.regresser = nn.Sequential(
            nn.Linear(num_feature * 4 * 4, num_feature*4*2),
            nn.LeakyReLU(0.2,inplace=True),
            nn.Linear(num_feature*4*2, num_feature*4),
            nn.LeakyReLU(0.2,inplace=True),
            nn.Linear(num_feature * 4, num_feature * 2),
            nn.LeakyReLU(0.2,inplace=True),
            nn.Linear(num_feature*2, num_feature),
            nn.LeakyReLU(0.2,inplace=True),
            nn.Linear(num_feature, num_points),
            # nn.Sigmoid()
        )
        self.num_feature = num_feature

    def Shuffle(self, x, y, random = None, int=int):
        """x, random=random.random -> shuffle list x in place; return None.
        Optional arg random is a 0-argument function returning a random
        float in [0.0, 1.0); by default, the standard random.random.
        """
        if random is None:
            random = self.random  # random=random.random
        # 转成numpy
        if torch.is_tensor(x) == True:
            if self.use_cuda == True:
                x = x.cpu().numpy()
            else:
                x = x.numpy()
        if torch.is_tensor(y) == True:
            if self.use_cuda == True:
                y = y.cpu().numpy()
            else:
                y = y.numpy()
        # 开始随机置换
        for i in range(len(x)):
            j = int(random() * (i + 1))
            if j <= len(x) - 1:  # 交换
                x[i], x[j] = x[j], x[i]
                y[i], y[j] = y[j], y[i]
        # 转回tensor
        if self.use_cuda == True:
            x = torch.from_numpy(x).cuda()
            y = torch.from_numpy(y).cuda()
        else:
            x = torch.from_numpy(x)
            y = torch.from_numpy(y)
        return x, y

    def forward(self, x):
        features = self.features(x)
        # print("features.size = ", features.size())
        features = features.view(-1,self.num_feature * 4 * 4)
        out = self.regresser(features)
        return out
