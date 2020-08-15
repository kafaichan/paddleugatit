import paddle
import paddle.fluid as fluid
from paddle.fluid.dygraph import Conv2D, Linear
import numpy as np
from paddle.fluid.dygraph.learning_rate_scheduler import LearningRateDecay


class ReflectionPad2d(fluid.dygraph.Layer):
    def __init__(self, pad):
        super(ReflectionPad2d, self).__init__()
        self.pad = pad

    def forward(self, inputs):
        out = fluid.layers.pad2d(input=inputs, paddings=[self.pad, self.pad, self.pad, self.pad], mode='reflect')
        return out


class ReLU(fluid.dygraph.Layer):
    def __init__(self):
        super(ReLU, self).__init__()

    def forward(self, inputs):
        return fluid.layers.relu(x=inputs)


class Upsample(fluid.dygraph.Layer):
    def __init__(self, scale_factor, mode):
        super(Upsample, self).__init__()
        self.scale_factor = scale_factor
        self.mode = mode
    
    def forward(self, inputs):
        return fluid.layers.image_resize(input=inputs, scale=self.scale_factor, resample=self.mode)


class Tanh(fluid.dygraph.Layer):
    def __init__(self):
        super(Tanh, self).__init__()

    def forward(self, inputs):
        return fluid.layers.tanh(x=inputs)


class LeakyReLU(fluid.dygraph.Layer):
    def __init__(self, alpha):
        super(LeakyReLU, self).__init__()
        self.alpha = alpha

    def forward(self, inputs):
        return fluid.layers.leaky_relu(x=inputs, alpha=self.alpha)


class SpectralNorm(fluid.dygraph.Layer):
    def __init__(self,
                 layer,
                 dim=0,
                 power_iters=1,
                 eps=1e-12,
                 dtype='float32'):
        super(SpectralNorm, self).__init__()
        self.spectral_norm_w = fluid.dygraph.SpectralNorm(layer.weight.shape, dim, power_iters, eps, dtype)
        self.dim = dim
        self.power_iters = power_iters
        self.eps = eps
        self.layer = layer

        weight = layer._parameters['weight']
        del layer._parameters['weight']
        self.weight_orig = self.create_parameter(weight.shape, dtype=weight.dtype)
        self.weight_orig.set_value(weight)

    def forward(self, x):
        weight = self.spectral_norm_w(self.weight_orig)
        self.layer.weight = weight
        out = self.layer(x)        
        return out


class L1Loss():
    def __init__(self, reduction='mean'):
        super(L1Loss, self).__init__()
        self.reduction = reduction

    def __call__(self, inputs, labels):
        if self.reduction == 'mean':
            diff = fluid.layers.abs(inputs-labels)
            return fluid.layers.reduce_mean(diff)


class BCEWithLogitsLoss():
    def __init__(self, weight=None, reduction='mean'):
        super(BCEWithLogitsLoss, self).__init__()
        self.weight = weight
        self.reduction = 'mean'

    def __call__(self, x, label):
        out = fluid.layers.sigmoid_cross_entropy_with_logits(x, label)
        if self.reduction == 'sum':
            return fluid.layers.reduce_sum(out)
        elif self.reduction == 'mean':
            return fluid.layers.reduce_mean(out)
        else:
            return out


class LinearLrCoolDown(LearningRateDecay):
    def __init__(self, learning_rate, cooldown_steps, end_lr=0, begin=1, step=1, dtype='float32'):
        super(LinearLrCoolDown, self).__init__(begin, step, dtype)
        type_check = isinstance(learning_rate, float) or isinstance(learning_rate, int)
        if not type_check:
            raise TypeError(
                "the type of learning_rate should be [int, float], the current type is {}".
                format(learning_rate))
        self.learning_rate = learning_rate
        self.cooldown_steps = cooldown_steps
        self.cooldown_lr_ratio = (learning_rate -float(end_lr)) / float(cooldown_steps)

    def step(self):
        if self.step_num <= self.cooldown_steps:
            return self.learning_rate
        else:
            return self.learning_rate - self.cooldown_lr_ratio * (self.step_num-self.cooldown_steps)