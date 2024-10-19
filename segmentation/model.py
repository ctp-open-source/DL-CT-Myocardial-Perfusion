import torch.nn as nn
import numpy as np

from segmentation.module.input import InputBlock
from segmentation.module.down import DownBlock
from segmentation.module.up import UpBlock
from segmentation.module.output import OutputBlock


class VBNet(nn.Module):
    """ VB-Net """

    def __init__(self, in_channels, out_channels):
        super(VBNet, self).__init__()

        self.in_block = InputBlock(in_channels, 16)
        self.down_32 = DownBlock(16, 1, use_bottle_neck=False)
        self.down_64 = DownBlock(32, 2, use_bottle_neck=True)
        self.down_128 = DownBlock(64, 3, use_bottle_neck=True)
        self.down_256 = DownBlock(128, 3, use_bottle_neck=True)
        self.up_256 = UpBlock(256, 256, 3, use_bottle_neck=True)
        self.up_128 = UpBlock(256, 128, 3, use_bottle_neck=True)
        self.up_64 = UpBlock(128, 64, 2, use_bottle_neck=False)
        self.up_32 = UpBlock(64, 32, 1, use_bottle_neck=False)
        self.out_block = OutputBlock(32, out_channels)

    def forward(self, input):
        print(input.shape)
        out16 = self.in_block(input)
        out32 = self.down_32(out16)
        out64 = self.down_64(out32)
        out128 = self.down_128(out64)
        out256 = self.down_256(out128)

        out = self.up_256(out256, out128)
        out = self.up_128(out, out64)
        out = self.up_64(out, out32)
        out = self.up_32(out, out16)
        out = self.out_block(out)
        return out

    def max_stride(self):
        return [16, 16, 16]


if __name__ == "__main__":
    import torch
    from thop import profile
    model = VBNet(1, 2)
    inputs = torch.randn(1, 1, 1, 128, 128, 128)
    flops, params = profile(model, inputs=inputs)
    print(flops, params)
