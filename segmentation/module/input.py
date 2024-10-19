import torch.nn as nn


class InputBlock(nn.Module):
    """ input block of VB-Net """

    def __init__(self, in_channels, out_channels):
        super(InputBlock, self).__init__()
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1)
        self.bn = nn.BatchNorm3d(out_channels)
        self.act = nn.ReLU(inplace=True)

    def forward(self, input):
        print("org shape", input.shape)
        out = self.act(self.bn(self.conv(input)))
        return out