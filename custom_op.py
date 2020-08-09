import paddle
import paddle.fluid as fluid
from paddle.fluid.dygraph import Conv2D, Pool2D, BatchNorm, Linear, Dropout
import numpy as np
import copy

class ReflectionPad2d(fluid.dygraph.Layer):
    def __init__(self, pad_size):
        super(ReflectionPad2d, self).__init__()
        self.pad_size = pad_size

    def forward(self, inputs):
        x = inputs.numpy()
        out = np.pad(x, 
            pad_width=((0,0), (0,0), (self.pad_size,self.pad_size), (self.pad_size, self.pad_size)), 
            mode='reflect')
        return fluid.dygraph.to_variable(out)


class SpectralNormConv2D(fluid.dygraph.Layer):
    def __init__(self, num_channels, num_filters, filter_size, stride, padding, bias_attr):
        super(SpectralNormConv2D, self).__init__()
        self.num_channels = num_channels
        self.num_filters = num_filters
        self.filter_size = filter_size
        self.stride = stride
        self.padding = padding
        self.bias_attr = bias_attr
        self.conv = Conv2D(num_channels, num_filters, filter_size, stride, padding, bias_attr)

    def forward(self, inputs):
        norm_weight = fluid.layers.spectral_norm(self.conv.weight).numpy()
        self.conv.weight.set_value(norm_weight)

        return self.conv(inputs)


class SpectralNormLinear(fluid.dygraph.Layer):
    def __init__(self, in_dim, out_dim, bias_attr):
        super(SpectralNormLinear, self).__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.bias_attr = bias_attr
        self.linear = Linear(in_dim, out_dim, bias_attr=bias_attr)

    def forward(self, inputs):
        norm_weight = fluid.layers.spectral_norm(self.linear.weight).numpy()

        self.linear.weight.set_value(norm_weight)

        return self.linear(inputs)
