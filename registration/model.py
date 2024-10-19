import torch
import torch.nn as nn
import torch.nn.functional as nnf

# Code snippet from voxelmorph
# Copyright (c) 2023 ahoopes
# Original source: https://github.com/voxelmorph/voxelmorph/blob/dev/voxelmorph/torch/losses.py


class SpatialTransformer(nn.Module):
    """
    N-D Spatial Transformer
    """

    def __init__(self, size, mode='bilinear'):
        super().__init__()

        self.mode = mode

        # create sampling grid
        vectors = [torch.arange(0, s) for s in size]
        grids = torch.meshgrid(vectors)
        grid = torch.stack(grids)
        grid = torch.unsqueeze(grid, 0)
        grid = grid.type(torch.FloatTensor)

        # registering the grid as a buffer cleanly moves it to the GPU, but it also
        # adds it to the state dict. this is annoying since everything in the state dict
        # is included when saving weights to disk, so the model files are way bigger
        # than they need to be. so far, there does not appear to be an elegant solution.
        # see: https://discuss.pytorch.org/t/how-to-register-buffer-without-polluting-state-dict
        self.register_buffer('grid', grid)

    def forward(self, src, flow):
        # new locations
        new_locs = self.grid + flow
        shape = flow.shape[2:]

        # need to normalize grid values to [-1, 1] for resampler
        for i in range(len(shape)):
            new_locs[:, i, ...] = 2 * (new_locs[:, i, ...] / (shape[i] - 1) - 0.5)

        # move channels dim to last position
        # also not sure why, but the channels need to be reversed
        if len(shape) == 2:
            new_locs = new_locs.permute(0, 2, 3, 1)
            new_locs = new_locs[..., [1, 0]]
        elif len(shape) == 3:
            new_locs = new_locs.permute(0, 2, 3, 4, 1)
            new_locs = new_locs[..., [2, 1, 0]]

        return nnf.grid_sample(src, new_locs, align_corners=True, mode=self.mode)

# End of snippet


class UNet3D(nn.Module):
    def __init__(self):
        super(UNet3D, self).__init__()
        self.down_block_1 = nn.Sequential(
            nn.Conv3d(2, 128, kernel_size=3, stride=1, padding=0),
            nn.BatchNorm3d(128),
            nn.ReLU(inplace=True),
            nn.Conv3d(128, 128, kernel_size=3, stride=1, padding=0),
            nn.BatchNorm3d(128),
            nn.ReLU(inplace=True)
        )
        self.pool_1 = nn.MaxPool3d(kernel_size=2, stride=2, padding=0)
        self.down_block_2 = nn.Sequential(
            nn.Conv3d(128, 256, kernel_size=3, stride=1, padding=0),
            nn.BatchNorm3d(256),
            nn.ReLU(inplace=True),
            nn.Conv3d(256, 256, kernel_size=3, stride=1, padding=0),
            nn.BatchNorm3d(256),
            nn.ReLU(inplace=True)
        )
        self.pool_2 = nn.MaxPool3d(kernel_size=2, stride=2, padding=0)
        self.down_block_3 = nn.Sequential(
            nn.Conv3d(256, 512, kernel_size=3, stride=1, padding=0),
            nn.BatchNorm3d(512),
            nn.ReLU(inplace=True),
            nn.Conv3d(512, 512, kernel_size=3, stride=1, padding=0),
            nn.BatchNorm3d(512),
            nn.ReLU(inplace=True)
        )
        self.up_block_1 = nn.Sequential(
            nn.ConvTranspose3d(512, 256, kernel_size=2, stride=2, padding=0),
            nn.BatchNorm3d(256),
            nn.ReLU(inplace=True)
        )
        self.up_block_2 = nn.Sequential(
            nn.Conv3d(512, 256, kernel_size=3, stride=1, padding=0),
            nn.BatchNorm3d(256),
            nn.ReLU(inplace=True),
            nn.Conv3d(256, 256, kernel_size=3, stride=1, padding=0),
            nn.BatchNorm3d(256),
            nn.ReLU(inplace=True)
        )
        self.up_block_3 = nn.Sequential(
            nn.ConvTranspose3d(256, 128, kernel_size=2, stride=2, padding=0),
            nn.BatchNorm3d(128),
            nn.ReLU(inplace=True)
        )
        self.up_block_4 = nn.Sequential(
            nn.Conv3d(256, 128, kernel_size=3, stride=1, padding=0),
            nn.BatchNorm3d(128),
            nn.ReLU(inplace=True),
        )
        self.out_block = nn.Sequential(
            nn.Conv3d(128, 3, kernel_size=3, stride=1, padding=0),
        )

    def crop_tensor(self, tensor, target_tensor):
        target_size = target_tensor.size()[2]
        tensor_size = tensor.size()[2]
        delta = tensor_size - target_size
        delta = delta // 2
        return tensor[:, :, delta:tensor_size-delta, delta:tensor_size-delta, delta:tensor_size-delta]

    def forward(self, input):
        out_1 = self.down_block_1(input)
        out_2 = self.down_block_2(self.pool_1(out_1))
        out = self.down_block_3(self.pool_2(out_2))
        out = self.up_block_1(out)
        crop_out_2 = self.crop_tensor(out_2, out)
        out = torch.cat((out, crop_out_2), 1)
        out = self.up_block_3(self.up_block_2(out))
        crop_out_1 = self.crop_tensor(out_1, out)
        out = torch.cat((out, crop_out_1), 1)
        out = self.up_block_4(out)
        out = self.out_block(out)
        out = out.contiguous()
        return out


if __name__ == '__main__':
    model = UNet3D()
    inputs = torch.randn(1, 2, 100, 100, 100)
    out = model(inputs)
    print(out.shape)


