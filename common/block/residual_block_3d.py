import torch.nn as nn
from common.layer.conv_bn_relu_3d import ConvBnRelu3d


class ResidualBlock3d(nn.Module):
    """ residual block with variable number of convolutions """

    def __init__(self, channels, ksize, padding, num_convs):
        super(ResidualBlock3d, self).__init__()

        layers = []
        for i in range(num_convs):
            if i != num_convs - 1:
                layers.append(ConvBnRelu3d(channels, channels, ksize, padding, do_act=True))
            else:
                layers.append(ConvBnRelu3d(channels, channels, ksize, padding, do_act=False))

        self.ops = nn.Sequential(*layers)
        self.act = nn.ReLU(inplace=True)

    def forward(self, input):

        output = self.ops(input)
        return self.act(input + output)