import torch.nn as nn
from common.layer.conv_bn_relu_3d import ConvBnRelu3d


class BottConvBnRelu3d(nn.Module):
    """Bottle neck structure"""

    def __init__(self, channels, ratio, do_act=True, bias=True):
        super(BottConvBnRelu3d, self).__init__()
        self.conv1 = ConvBnRelu3d(channels, channels//ratio, ksize=1, padding=0, do_act=True, bias=bias)
        self.conv2 = ConvBnRelu3d(channels//ratio, channels//ratio, ksize=3, padding=1, do_act=True, bias=bias)
        self.conv3 = ConvBnRelu3d(channels//ratio, channels, ksize=1, padding=0, do_act=do_act, bias=bias)

    def forward(self, input):
        out = self.conv3(self.conv2(self.conv1(input)))
        return out
