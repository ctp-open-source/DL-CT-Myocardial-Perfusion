import torch.nn as nn
from common.block.bott_conv_bn_relu_3d import BottConvBnRelu3d


class BottResidualBlock3d(nn.Module):
    """ block with bottle neck conv"""

    def __init__(self, channels, ratio, num_convs):
        super(BottResidualBlock3d, self).__init__()
        layers = []
        for i in range(num_convs):
            if i != num_convs - 1:
                layers.append(BottConvBnRelu3d(channels, ratio, True))
            else:
                layers.append(BottConvBnRelu3d(channels, ratio, False))

        self.ops = nn.Sequential(*layers)
        self.act = nn.ReLU(inplace=True)

    def forward(self, input):
        output = self.ops(input)
        return self.act(input + output)
