import math

from torch import nn
import torch.nn.functional as F


def c2_xavier_fill(module):
    nn.init.kaiming_uniform_(module.weight, a=1)
    if module.bias is not None:
        nn.init.constant_(module.bias, 0)


class FPN(nn.Module):
    def __init__(self, in_channels, out_channels, top_block=None, fuse_type="sum"):
        super().__init__()

        lateral_convs = []
        output_convs = []

        for idx, in_channel in enumerate(in_channels):
            lateral_conv = nn.Conv2d(
                in_channel, out_channels, kernel_size=1, bias=True
            )
            output_conv = nn.Conv2d(
                out_channels,
                out_channels,
                kernel_size=3,
                stride=1,
                padding=1,
                bias=True
            )

            c2_xavier_fill(lateral_conv)
            c2_xavier_fill(output_conv)

            lateral_convs.append(lateral_conv)
            output_convs.append(output_conv)

        # Place convs into top-down order (from low to high resolution)
        # to make the top-down computation in forward clearer.
        self.lateral_convs = lateral_convs[::-1]
        self.output_convs = output_convs[::-1]

        self.top_block = top_block

        assert fuse_type in {"avg", "sum"}
        self._fuse_type = fuse_type

    def forward(self, x):

        results = []
        prev_features = self.lateral_convs[0](x[-1])
        for idx, (lateral_conv, output_conv) in enumerate(zip(self.lateral_convs, self.output_convs)):
            if idx == 0:
                prev_features = lateral_conv(x[-idx - 1])
                results.append(output_conv(prev_features))
            else:
                top_down_features = F.interpolate(prev_features, scale_factor=2.0, mode="nearest")
                features = lateral_conv(x[-idx - 1])
                prev_features = features + top_down_features
                if self._fuse_type == "avg":
                    prev_features /= 2
                results.insert(0, output_conv(prev_features))

        if self.top_block is not None:
            top_block_in_feature = results[-1]
            results.extend(self.top_block(top_block_in_feature))

        return results


class LastLevelMaxPool(nn.Module):
    """
    This module is used in the original FPN to generate a downsampled
    P6 feature from P5.
    """

    def __init__(self):
        super().__init__()
        self.num_levels = 1

    def forward(self, x):
        return [F.max_pool2d(x, kernel_size=1, stride=2, padding=0)]
