from __future__ import print_function

import torch
import torch.nn as nn
import torch.nn.functional as F

def downsample(x, p):
    for i in range(p):
        x = F.max_pool2d(x, kernel_size=2)
    return x


def upsample(x, p):
    x = F.interpolate(x, scale_factor=2 ** abs(p), mode='nearest')
    return x


def identity(x, p):
    return x


def skip(x):
    pass

def sampling_func(p):
    if p == 0:
        smp_func = identity
    elif p > 0:
        smp_func = downsample
    elif p < 0:
        smp_func = upsample
    return smp_func


def string_to_list(x):
    '''
    convert a string to a list of integers
    :param x: a string containing ","
    :return: a list of integers
    '''
    x = x.split(',')
    res = [int(i) for i in x]
    return res

class sConv2d(nn.Module):
    def __init__(self, in_, out_, kernel_size):
        '''
        Creates a layer of network by applying convolution over multiple resolutions and summing the results with weight alpha

        :param in_: number of input channels
        :param out_: number of output channels
        :param kernel_size: convolution filter size
        '''
        super(sConv2d, self).__init__()
        self.in_ = in_ # input channels
        self.out_ = out_ # number of channels for the output of the layer

        # convolution
        self.conv = nn.Conv2d(in_, out_, kernel_size, stride=1, padding=1, bias=False)
        self.relu = nn.ReLU()
        self.bn = nn.BatchNorm2d(out_, affine=False, track_running_stats=False)

    def forward(self, x):
        '''
        input: feature map of size (batch_size, in_, H, W)
        output: feature map of size (batch_size, out_ , H, W)
        '''
        y = self.conv(x)
        y = self.relu(y)
        out = self.bn(y)
        return out

class multires_model(nn.Module):
    def __init__(self, ncat=10, channels=16, leng=10, max_scales=4):
        '''
        create CNN by stacking layers
        :param ncat: number of classes
        :param channels: number of channels for first resolution
        :param leng: depth of the network
        :param max_scales: how many scales to use
        '''
        super(multires_model, self).__init__()
        self.leng = leng
        channels = int(channels)
        listc = [sConv2d(3, channels, 3)]
        listc += [sConv2d(channels, channels, 3) for i in range(leng - 2)]
        listc += [sConv2d(channels, ncat, 3)]
        self.conv = nn.ModuleList(listc)

    def set_path(self, path_id, relative_path):
        self.path_id = path_id
        self.relative_path = relative_path
        self.sample_functions = [sampling_func(i) for i in self.relative_path]

    def forward(self, x):
        '''
        input: RGB image batch of size (batch_size, 3, H, W)
        output: vector of class probabilities of size (batch_size, ncat)
        '''
        out = x
        for c in range(self.leng):
            out = self.sample_functions[c](out, self.relative_path[c])
            out = self.conv[c](out)
        out = F.avg_pool2d(out, kernel_size=out.size()[2:]) # Global average pooling
        out = out.view(out.size(0), -1)
        return out