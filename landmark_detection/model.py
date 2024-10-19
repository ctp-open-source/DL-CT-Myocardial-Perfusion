import torch
import torch.nn as nn

from common.block.residual_block_3d import ResidualBlock3d


class LandmarkDetectionNet(nn.Module):
    def __init__(self, in_channels, num_classes):
        super(LandmarkDetectionNet, self).__init__()
        self.in_block = nn.Sequential(
            nn.Conv3d(in_channels, 16, kernel_size=3, padding=1),
            nn.BatchNorm3d(16),
            nn.ReLU(inplace=True)
        )
        self.down_32 = nn.Sequential(
            nn.Conv3d(16, 32, kernel_size=(2, 2, 2), stride=(2, 2, 2), groups=1),
            nn.BatchNorm3d(32),
            nn.ReLU(inplace=True),
            ResidualBlock3d(32, 3, 1, 1)
        )
        self.down_64 = nn.Sequential(
            nn.Conv3d(32, 64, kernel_size=(2, 2, 2), stride=(2, 2, 2), groups=1),
            nn.BatchNorm3d(64),
            nn.ReLU(inplace=True),
            ResidualBlock3d(64, 3, 1, 2)
        )
        self.up_64_1 = nn.Sequential(
            nn.ConvTranspose3d(64, 32, kernel_size=(2, 2, 2), stride=(2, 2, 2), groups=1),
            nn.BatchNorm3d(32),
            nn.ReLU(inplace=True)
        )
        self.up_64_2 = ResidualBlock3d(64, 3, 1, 2)
        self.up_32_1 = nn.Sequential(
            nn.ConvTranspose3d(64, 16, kernel_size=(2, 2, 2), stride=(2, 2, 2), groups=1),
            nn.BatchNorm3d(16),
            nn.ReLU(inplace=True)
        )
        self.up_32_2 = ResidualBlock3d(32, 3, 1, 1)
        self.out_block = nn.Sequential(
            nn.Conv3d(32, num_classes, kernel_size=3, padding=1),
            nn.BatchNorm3d(num_classes),
            nn.ReLU(inplace=True),
            nn.Conv3d(num_classes, num_classes, kernel_size=1)
        )
        self.softmax = nn.Softmax(1)

    def forward(self, input):
        out16 = self.in_block(input)
        out32 = self.down_32(out16)
        out64 = self.down_64(out32)
        out = self.up_64_1(out64)
        out = torch.cat((out, out32), 1)
        out = self.up_64_2(out)
        out = self.up_32_1(out)
        out = torch.cat((out, out16), 1)
        out = self.up_32_2(out)
        out = self.out_block(out)
        out = self.softmax(out)
        out = out.contiguous()
        return out


if __name__ == '__main__':
    model = LandmarkDetectionNet(1, 3)
    inputs = torch.randn(1, 1, 100, 100, 100)
    out = model(inputs)
    print(out.shape)
