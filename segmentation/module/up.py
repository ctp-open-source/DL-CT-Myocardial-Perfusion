import torch
import torch.nn as nn

from common.block.bott_residual_block_3d import BottResidualBlock3d
from common.block.residual_block_3d import ResidualBlock3d


class UpBlock(nn.Module):
    """ Upsample block of VB-Net """

    def __init__(self, in_channels, out_channels, num_convs, use_bottle_neck=False, kernel_size=[2, 2, 2], stride=[2, 2, 2]):
        super(UpBlock, self).__init__()
        self.up_conv = nn.ConvTranspose3d(in_channels, out_channels // 2, kernel_size=kernel_size, stride=kernel_size, groups=1)
        self.up_bn = nn.BatchNorm3d(out_channels // 2)
        self.up_act = nn.ReLU(inplace=True)
        if use_bottle_neck:
            self.rblock = BottResidualBlock3d(out_channels, 4, num_convs)
        else:
            self.rblock = ResidualBlock3d(out_channels, 3, 1, num_convs)

    def forward(self, input, skip):
        out = self.up_act(self.up_bn(self.up_conv(input)))
        out = torch.cat((out, skip), 1)
        out = self.rblock(out)
        return out