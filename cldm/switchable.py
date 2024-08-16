import torch.nn as nn
from torch import Tensor


class SwitchableGroupNorm(nn.GroupNorm):
    def __init__(self, *args, norm_layer=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.norm_layer = norm_layer

    def set_norm_layer(self, norm_layer: nn.GroupNorm):
        self.norm_layer = norm_layer

    def copy_weights(self):
        if self.norm_layer is not None:
            self.norm_layer.weight.data.copy_(self.weight.data)
            self.norm_layer.bias.data.copy_(self.bias.data)

    def forward(self, x: Tensor):
        if self.norm_layer is None:
            return super().forward(x)
        return self.norm_layer(x)


class SwitchableLayerNorm(nn.LayerNorm):
    def __init__(self, *args, norm_layer=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.norm_layer = norm_layer

    def set_norm_layer(self, norm_layer: nn.LayerNorm):
        self.norm_layer = norm_layer

    def copy_weights(self):
        if self.norm_layer is not None:
            self.norm_layer.weight.data.copy_(self.weight.data)
            self.norm_layer.bias.data.copy_(self.bias.data)

    def forward(self, x: Tensor):
        if self.norm_layer is None:
            return super().forward(x)
        return self.norm_layer(x)


class SwitchableConv2d(nn.Conv2d):
    def __init__(self, *args, conv_layer=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.conv_layer = conv_layer

    def set_conv_layer(self, conv_layer: nn.Conv2d):
        self.conv_layer = conv_layer

    def copy_weights(self):
        if self.conv_layer is not None:
            self.conv_layer.weight.data.copy_(self.weight.data)
            if hasattr(self, 'bias') and self.bias is not None:
                self.conv_layer.bias.data.copy_(self.bias.data)

    def forward(self, x: Tensor):
        if self.conv_layer is None:
            return super().forward(x)
        return self.conv_layer(x)
