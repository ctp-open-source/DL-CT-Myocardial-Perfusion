import torch.nn as nn


class ConvBnRelu3d(nn.Module):
    """ 
    classic combination: 
    conv + batch normalization [+ relu]
    post-activation mode 
    """

    def __init__(self, in_channels, out_channels, ksize, padding, do_act=True, bias=True):
        super(ConvBnRelu3d, self).__init__()
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size=ksize, padding=padding, groups=1, bias=bias)
        self.bn = nn.BatchNorm3d(out_channels)
        self.do_act = do_act
        if do_act:
            self.act = nn.ReLU(inplace=True)

    def forward(self, input):
        out = self.bn(self.conv(input))
        if self.do_act:
            out = self.act(out)
        return out
