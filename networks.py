from custom_op import *
import paddle.fluid as fluid
from paddle.fluid.dygraph import Conv2D, Linear, to_variable, InstanceNorm
from paddle.fluid.initializer import ConstantInitializer
import numpy as np


class ResnetGenerator(fluid.dygraph.Layer):
    def __init__(self, input_nc, output_nc, ngf=64, n_blocks=6, img_size=256, light=False):
        assert(n_blocks >= 0)
        super(ResnetGenerator, self).__init__()
        self.input_nc = input_nc
        self.output_nc = output_nc
        self.ngf = ngf
        self.n_blocks = n_blocks
        self.img_size = img_size
        self.light = light

        DownBlock = []
        DownBlock += [ReflectionPad2d(3),
                      Conv2D(num_channels=input_nc, num_filters=ngf, filter_size=7, stride=1, padding=0, bias_attr=False),
                      InstanceNorm(ngf),
                      ReLU()]

        # Down-Sampling
        n_downsampling = 2
        for i in range(n_downsampling):
            mult = 2**i
            DownBlock += [ReflectionPad2d(1),
                          Conv2D(num_channels=ngf*mult, num_filters=ngf*mult*2, filter_size=3, stride=2, padding=0, bias_attr=False),
                          InstanceNorm(ngf * mult * 2),
                          ReLU()]

        # Down-Sampling Bottleneck
        mult = 2**n_downsampling
        for i in range(n_blocks):
            DownBlock += [ResnetBlock(ngf*mult, use_bias=False)]

        # Class Activation Map
        self.gap_fc = Linear(ngf*mult, 1, bias_attr=False)
        self.gmp_fc = Linear(ngf*mult, 1, bias_attr=False)
        self.conv1x1 = Conv2D(ngf*mult*2, ngf*mult, filter_size=1, stride=1, bias_attr=True)
        self.relu = ReLU()

        # Gamma, Beta block
        if self.light:
            FC = [Linear(ngf * mult, ngf * mult, bias_attr=False, act='relu'),
                  Linear(ngf * mult, ngf * mult, bias_attr=False, act='relu')]
        else:
            FC = [Linear(img_size // mult * img_size // mult * ngf * mult, ngf * mult, bias_attr=False, act='relu'),
                  Linear(ngf * mult, ngf * mult, bias_attr=False, act='relu')]
        self.gamma = Linear(ngf * mult, ngf * mult, bias_attr=False)
        self.beta = Linear(ngf * mult, ngf * mult, bias_attr=False)

        # Up-Sampling Bottleneck
        for i in range(n_blocks):
            setattr(self, 'UpBlock1_' + str(i+1), ResnetAdaILNBlock(ngf * mult, use_bias=False))

        # Up-Sampling
        UpBlock2 = []
        for i in range(n_downsampling):
            mult = 2**(n_downsampling - i)
            UpBlock2 += [Upsample(scale_factor=2, mode='NEAREST'),
                         ReflectionPad2d(1),
                         Conv2D(ngf * mult, int(ngf * mult / 2), filter_size=3, stride=1, padding=0, bias_attr=False),
                         ILN(int(ngf * mult / 2)),
                         ReLU()]

        UpBlock2 += [ReflectionPad2d(3),
                     Conv2D(ngf, output_nc, filter_size=7, stride=1, padding=0, bias_attr=False),
                     Tanh()]

        self.DownBlock = fluid.dygraph.Sequential(*DownBlock)
        self.FC = fluid.dygraph.Sequential(*FC)
        self.UpBlock2 = fluid.dygraph.Sequential(*UpBlock2)

    def forward(self, inputs):
        x = self.DownBlock(inputs)

        gap = fluid.layers.adaptive_pool2d(input=x, pool_size=1, pool_type='avg')
        gap_logit = self.gap_fc(fluid.layers.reshape(x=gap, shape=(x.shape[0], -1)))
        gap = x * fluid.layers.unsqueeze(fluid.layers.transpose(self.gap_fc.weight, perm=[1,0]), axes=[2,3])

        gmp = fluid.layers.adaptive_pool2d(input=x, pool_size=1, pool_type='max')
        gmp_logit = self.gmp_fc(fluid.layers.reshape(x=gmp, shape=(x.shape[0], -1)))
        gmp = x * fluid.layers.unsqueeze(fluid.layers.transpose(self.gmp_fc.weight, perm=[1,0]), axes=[2,3])

        cam_logit = fluid.layers.concat(input=[gap_logit, gmp_logit], axis=1)
        x = fluid.layers.concat(input=[gap, gmp], axis=1)
        x = self.relu(self.conv1x1(x))
        heatmap = fluid.layers.reduce_sum(x, dim=1, keep_dim=True)

        if self.light:
            x_ = fluid.layers.adaptive_pool2d(input=x, pool_size=1, pool_type='avg')
            x_ = self.FC(fluid.layers.reshape(x=x_, shape=(x_.shape[0], -1)))
        else:
            x_ = self.FC(fluid.layers.reshape(x=x, shape=(x.shape[0], -1)))

        gamma, beta = self.gamma(x_), self.beta(x_)

        for i in range(self.n_blocks):
            x = getattr(self, 'UpBlock1_' + str(i+1))(x, gamma, beta)

        out = self.UpBlock2(x)
        return out, cam_logit, heatmap

    def clip_rho(self, vmin=0, vmax=1):
        for name, param in self.named_parameters():
            if 'rho' in name:
                param.set_value(fluid.layers.clip(param, vmin, vmax))


class ResnetBlock(fluid.dygraph.Layer):
    def __init__(self, dim, use_bias):
        super(ResnetBlock, self).__init__()
        conv_block = []
        conv_block += [ReflectionPad2d(1),
                       Conv2D(num_channels=dim, num_filters=dim, filter_size=3, stride=1, padding=0, bias_attr=use_bias),
                       InstanceNorm(dim),
                       ReLU()]

        conv_block += [ReflectionPad2d(1),
                       Conv2D(num_channels=dim, num_filters=dim, filter_size=3, stride=1, padding=0, bias_attr=use_bias),
                       InstanceNorm(dim)]

        self.conv_block = fluid.dygraph.Sequential(*conv_block)

    def forward(self, x):
        out = x + self.conv_block(x)
        return out


class ResnetAdaILNBlock(fluid.dygraph.Layer):
    def __init__(self, dim, use_bias):
        super(ResnetAdaILNBlock, self).__init__()
        self.dim = dim
        self.use_bias = use_bias
        self.norm1 = adaILN(dim)
        self.norm2 = adaILN(dim)

    def forward(self, x, gamma, beta):
        out = fluid.layers.pad2d(input=x, paddings=[1,1,1,1], mode='reflect')
        out = fluid.layers.conv2d(input=out, num_filters=self.dim, filter_size=3, stride=1, padding=0, groups=1, bias_attr=self.use_bias)
        out = self.norm1(out, gamma, beta)
        out = fluid.layers.relu(x=out)
        out = fluid.layers.pad2d(input=out, paddings=[1,1,1,1], mode='reflect')
        out = fluid.layers.conv2d(input=out, num_filters=self.dim, filter_size=3, stride=1, padding=0, groups=1, bias_attr=self.use_bias)
        out = self.norm2(out, gamma, beta)
        return out + x


class adaILN(fluid.dygraph.Layer):
    def __init__(self, num_features, eps=1e-5):
        super(adaILN, self).__init__()
        self.eps = eps
        self.add_parameter('rho', fluid.layers.create_parameter(
            shape=[1, num_features, 1, 1], dtype='float32',
            default_initializer=ConstantInitializer(0.9)
            ))

    def forward(self, inputs, gamma, beta):
        in_mean = fluid.layers.reduce_mean(input=inputs, dim=[2,3], keep_dim=True)
        in_var = fluid.layers.reduce_mean(fluid.layers.pow(inputs-in_mean, 2.0), dim=[2,3], keep_dim=True)
        out_in = (inputs-in_mean) / fluid.layers.sqrt(in_var+self.eps)

        ln_mean = fluid.layers.reduce_mean(input=inputs, dim=[1,2,3], keep_dim=True)
        ln_var = fluid.layers.reduce_mean(fluid.layers.pow(inputs-ln_mean, 2.0), dim=[1,2,3], keep_dim=True)
        out_ln = (inputs-ln_mean) / fluid.layers.sqrt(ln_var + self.eps)

        out = self.rho*out_in +  (1-self.rho)*out_ln
        out = out * fluid.layers.unsqueeze(gamma, axes=[2,3]) + fluid.layers.unsqueeze(beta, axes=[2,3])
        return out


class ILN(fluid.dygraph.Layer):
    def __init__(self, num_features, eps=1e-5):
        super(ILN, self).__init__()
        self.eps = eps
        self.add_parameter('rho', fluid.layers.create_parameter(
            shape=[1, num_features, 1, 1], dtype='float32',
            default_initializer=ConstantInitializer(0.0)))
        self.add_parameter('gamma', fluid.layers.create_parameter(
            shape=[1, num_features, 1, 1], dtype='float32',
            default_initializer=ConstantInitializer(1.0)))
        self.add_parameter('beta', fluid.layers.create_parameter(
            shape=[1, num_features, 1, 1], dtype='float32',
            default_initializer=ConstantInitializer(0.0)))

    def forward(self, inputs):
        in_mean = fluid.layers.reduce_mean(input=inputs, dim=[2,3], keep_dim=True)
        in_var = fluid.layers.reduce_mean(fluid.layers.pow(inputs-in_mean, 2.0), dim=[2,3], keep_dim=True)
        out_in = (inputs - in_mean) / fluid.layers.sqrt(in_var + self.eps)

        ln_mean = fluid.layers.reduce_mean(input=inputs, dim=[1,2,3], keep_dim=True)
        ln_var = fluid.layers.reduce_mean(fluid.layers.pow(inputs-ln_mean, 2.0), dim=[1,2,3], keep_dim=True)
        out_ln = (inputs - ln_mean) / fluid.layers.sqrt(ln_var + self.eps)

        out = self.rho*out_in + (1-self.rho)*out_ln
        out = out*self.gamma + self.beta
        return out


class Discriminator(fluid.dygraph.Layer):
    def __init__(self, input_nc, ndf=64, n_layers=5):
        super(Discriminator, self).__init__()
        model = [ReflectionPad2d(1),
                 SpectralNorm(Conv2D(input_nc, ndf, filter_size=4, stride=2, padding=0, bias_attr=True)),
                 LeakyReLU(0.2)]

        for i in range(1, n_layers - 2):
            mult = 2 ** (i - 1)
            model += [ReflectionPad2d(1),
                      SpectralNorm(Conv2D(ndf * mult, ndf * mult * 2, filter_size=4, stride=2, padding=0, bias_attr=True)),
                      LeakyReLU(0.2)]

        mult = 2 ** (n_layers - 2 - 1)
        model += [ReflectionPad2d(1),
                  SpectralNorm(Conv2D(ndf * mult, ndf * mult * 2, filter_size=4, stride=1, padding=0, bias_attr=True)),
                  LeakyReLU(0.2)]

        # Class Activation Map
        mult = 2 ** (n_layers - 2)
        self.gap_fc = SpectralNorm(Linear(ndf * mult, 1, bias_attr=False))
        self.gmp_fc = SpectralNorm(Linear(ndf * mult, 1, bias_attr=False))
        self.conv1x1 = Conv2D(ndf * mult * 2, ndf * mult, filter_size=1, stride=1, bias_attr=True)
        self.leaky_relu = LeakyReLU(0.2)

        self.pad = ReflectionPad2d(1)
        self.conv = SpectralNorm(Conv2D(ndf * mult, 1, filter_size=4, stride=1, padding=0, bias_attr=False))

        self.model = fluid.dygraph.Sequential(*model)

    def forward(self, inputs):
        x = self.model(inputs)
        gap = fluid.layers.adaptive_pool2d(x, pool_size=1, pool_type='avg')
        gap_logit = self.gap_fc(fluid.layers.reshape(x=gap, shape=(x.shape[0], -1)))
        gap = x * fluid.layers.unsqueeze(input=fluid.layers.transpose(self.gap_fc.layer.weight, perm=[1,0]), axes=[2,3])

        gmp = fluid.layers.adaptive_pool2d(x, pool_size=1, pool_type='max')
        gmp_logit = self.gmp_fc(fluid.layers.reshape(x=gmp, shape=(x.shape[0], -1)))
        gmp = x * fluid.layers.unsqueeze(input=fluid.layers.transpose(self.gmp_fc.layer.weight, perm=[1,0]), axes=[2,3])

        cam_logit = fluid.layers.concat([gap_logit, gmp_logit], 1)
        x = fluid.layers.concat([gap, gmp], 1)
        x = self.leaky_relu(self.conv1x1(x))

        heatmap = fluid.layers.reduce_sum(x, dim=1, keep_dim=True)

        x = self.pad(x)
        out = self.conv(x)

        return out, cam_logit, heatmap