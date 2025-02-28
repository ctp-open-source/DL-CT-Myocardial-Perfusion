import torch.nn as nn

from common.block.bott_residual_block_3d import BottResidualBlock3d
from common.block.residual_block_3d import ResidualBlock3d


class DownBlock(nn.Module):
    """ downsample block of VB-Net """

    def __init__(
        self, 
        in_channels, 
        num_convs, 
        use_bottle_neck=False, 
        kernel_size=[2, 2, 2], 
        stride=[2, 2, 2]
    ):
        super(DownBlock, self).__init__()
        out_channels = in_channels * 2
        self.down_conv = nn.Conv3d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, groups=1)
        self.down_bn = nn.BatchNorm3d(out_channels)
        self.down_act = nn.ReLU(inplace=True)
        if use_bottle_neck:
            self.rblock = BottResidualBlock3d(out_channels, 4, num_convs)
        else:
            self.rblock = ResidualBlock3d(out_channels, 3, 1, num_convs)

    def forward(self, input):
        out = self.down_act(self.down_bn(self.down_conv(input)))
        out = self.rblock(out)
        return out