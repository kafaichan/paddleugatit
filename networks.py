from custom_op import SpectralNormConv2D, SpectralNormLinear
import paddle.fluid as fluid
from paddle.fluid.dygraph import Conv2D, Pool2D, BatchNorm, Linear, Dropout, to_variable
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

        self.n_downsampling = 2
        mult = 2**self.n_downsampling
        

    def forward(self, inputs):
        x = fluid.layers.pad2d(input=inputs, paddings=[3,3,3,3], mode='reflect')
        x = fluid.layers.conv2d(input=x, num_filters=self.ngf, filter_size=7, stride=1, padding=0, groups=1, bias_attr=False)
        x = fluid.layers.instance_norm(input=x)
        x = fluid.layers.relu(x=x)

        # Down-Sampling
        for i in range(self.n_downsampling):
            mult = 2**i
            x = fluid.layers.pad2d(input=x, paddings=[1,1,1,1], mode='reflect')
            x = fluid.layers.conv2d(input=x, num_filters=self.ngf*mult*2, filter_size=3, stride=2, padding=0, groups=1, bias_attr=False)
            x = fluid.layers.instance_norm(input=x)
            x = fluid.layers.relu(x=x)
        
        # Down-Sampling Bottleneck
        mult = 2**self.n_downsampling
        for i in range(self.n_blocks):
            x = ResnetBlock(self.ngf*mult, use_bias=False)(x)

        gap = fluid.layers.adaptive_pool2d(input=x, pool_size=1, pool_type='avg')
        self.gap_linear = Linear(input_dim=self.ngf*mult, output_dim=1, bias_attr=False)
        gap_logit = self.gap_linear(fluid.layers.reshape(x=gap, shape=[x.shape[0], -1]))
        gap_weight = fluid.layers.transpose(list(self.gap_linear.parameters())[0], perm=[1,0])
        gap = x * fluid.layers.unsqueeze(gap_weight, axes=[2,3])


        gmp = fluid.layers.adaptive_pool2d(input=x, pool_size=1, pool_type='max')
        self.gmp_linear = Linear(input_dim=self.ngf*mult, output_dim=1, bias_attr=False)
        gmp_logit = self.gmp_linear(fluid.layers.reshape(x=gmp, shape=[x.shape[0], -1]))
        gmp_weight = fluid.layers.transpose(list(self.gmp_linear.parameters())[0], perm=[1,0])
        gmp = x * fluid.layers.unsqueeze(gmp_weight, axes=[2,3])

        cam_logit = fluid.layers.concat(input=[gap_logit, gmp_logit], axis=1)
        x = fluid.layers.concat(input=[gap, gmp], axis=1)
        self.conv1x1 = Conv2D(num_channels=self.ngf*mult*2, num_filters=self.ngf*mult, filter_size=1, stride=1, bias_attr=False)
        x = fluid.layers.relu(self.conv1x1(x))

        heatmap = fluid.layers.reduce_sum(x, dim=1, keep_dim=True)

        if self.light:
            x_ = fluid.layers.adaptive_pool2d(input=x, pool_size=1, pool_type='avg')
            x_ = fluid.layers.fc(input=x_, size=self.ngf*mult, bias_attr=False)
            x_ = fluid.layers.relu(x=x_)
            x_ = fluid.layers.fc(input=x_, size=self.ngf*mult, bias_attr=False)
            x_ = fluid.layers.relu(x=x_)
        else:
            x_ = fluid.layers.fc(input=x, size=self.ngf*mult, bias_attr=False)
            x_ = fluid.layers.relu(x=x_)
            x_ = fluid.layers.fc(input=x_, size=self.ngf*mult, bias_attr=False)
            x_ = fluid.layers.relu(x=x_)
        gamma = fluid.layers.fc(input=x_, size=self.ngf*mult, bias_attr=False)
        beta = fluid.layers.fc(input=x_, size=self.ngf*mult, bias_attr=False)

        for i in range(self.n_blocks):
            x = ResnetAdaILNBlock(self.ngf*mult, use_bias=False)(x, gamma, beta)
        
        # Upblock 2
        for i in range(self.n_downsampling):
            mult = 2 ** (self.n_downsampling-i)
            x = fluid.layers.image_resize(input=x, scale=2, resample='NEAREST')
            x = fluid.layers.pad2d(input=x, paddings=[1,1,1,1], mode='reflect')
            x = fluid.layers.conv2d(input=x, num_filters=int(self.ngf*mult/2), filter_size=3, stride=1, padding=0, groups=1, bias_attr=False)
            x = ILN(int(self.ngf*mult/2))(x)
            x = fluid.layers.relu(x=x)

        x = fluid.layers.pad2d(input=x, paddings=[3,3,3,3], mode='reflect')
        x = fluid.layers.conv2d(input=x, num_filters=self.output_nc, filter_size=7, stride=1, padding=0, groups=1, bias_attr=False)
        out = fluid.layers.tanh(x=x)

        cam_logit = fluid.layers.sigmoid(x=cam_logit)
        return out, cam_logit, heatmap

class ResnetBlock(fluid.dygraph.Layer):
    def __init__(self, dim, use_bias):
        super(ResnetBlock, self).__init__()
        self.dim = dim
        self.use_bias = use_bias

    def forward(self, inputs):
        x = fluid.layers.pad2d(input=inputs, paddings=[1,1,1,1], mode='reflect')
        x = fluid.layers.conv2d(input=x, num_filters=self.dim, filter_size=3, stride=1, padding=0, groups=1, bias_attr=self.use_bias)
        x = fluid.layers.instance_norm(input=x)
        x = fluid.layers.relu(x=x)
        x = fluid.layers.pad2d(input=x, paddings=[1,1,1,1], mode='reflect')
        x = fluid.layers.conv2d(input=x, num_filters=self.dim, filter_size=3, stride=1, padding=0, groups=1, bias_attr=self.use_bias)
        x = fluid.layers.instance_norm(input=x)
        out = inputs + x
        return out


class ResnetAdaILNBlock(fluid.dygraph.Layer):
    def __init__(self, dim, use_bias):
        super(ResnetAdaILNBlock, self).__init__()
        self.conv1 = Conv2D(dim, dim, filter_size=3, stride=1, padding=0, bias_attr=use_bias)
        self.norm1 = adaILN(dim)
        self.conv2 = Conv2D(dim, dim, filter_size=3, stride=1, padding=0, bias_attr=use_bias)
        self.norm2 = adaILN(dim)

    def forward(self, x, gamma, beta):
        out = fluid.layers.pad2d(input=x, paddings=[1,1,1,1], mode='reflect')
        out = self.conv1(out)
        out = self.norm1(out, gamma, beta)
        out = fluid.layers.relu(x=out)
        out = fluid.layers.pad2d(input=out, paddings=[1,1,1,1], mode='reflect')
        out = self.conv2(out)
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
        in_var = to_variable(np.var(inputs.numpy(), axis=(2,3), keepdims=True))
        out_in = (inputs-in_mean) / fluid.layers.sqrt(in_var+self.eps)

        ln_mean = fluid.layers.reduce_mean(input=inputs, dim=[1,2,3], keep_dim=True)
        ln_var = to_variable(np.var(inputs.numpy(), axis=(1,2,3), keepdims=True))
        out_ln = (inputs-ln_mean) / fluid.layers.sqrt(ln_var + self.eps)

        out = fluid.layers.expand(x=self.rho, expand_times=[inputs.shape[0], 1, inputs.shape[2], inputs.shape[3]]) * out_in + \
              fluid.layers.expand(x=(1-self.rho), expand_times=[inputs.shape[0], 1, inputs.shape[2], inputs.shape[3]]) * out_ln
        out = out * fluid.layers.unsqueeze(gamma, axes=[2,3]) + fluid.layers.unsqueeze(beta, axes=[2,3])
        return out
    
    def clip_rho(self, rho_clipper):
        self.rho.set_value(rho_clipper(self.rho))


class ILN(fluid.dygraph.Layer):
    def __init__(self, num_features, eps=1e-5):
        super(ILN, self).__init__()
        self.eps = eps
        self.add_parameter('rho', fluid.layers.create_parameter(
            shape=[1, num_features, 1, 1], dtype='float32',#))
            default_initializer=ConstantInitializer(0.0)))
        self.add_parameter('gamma', fluid.layers.create_parameter(
            shape=[1, num_features, 1, 1], dtype='float32',#))
            default_initializer=ConstantInitializer(1.0)))
        self.add_parameter('beta', fluid.layers.create_parameter(
            shape=[1, num_features, 1, 1], dtype='float32',#))
            default_initializer=ConstantInitializer(0.0)))

    def forward(self, inputs):
        in_mean = fluid.layers.reduce_mean(input=inputs, dim=[2,3], keep_dim=True)
        in_var = to_variable(np.var(inputs.numpy(), axis=(2,3), keepdims=True))
        out_in = (inputs - in_mean) / fluid.layers.sqrt(in_var + self.eps)

        ln_mean = fluid.layers.reduce_mean(input=inputs, dim=[1,2,3], keep_dim=True)
        ln_var = to_variable(np.var(inputs.numpy(), axis=(1,2,3), keepdims=True))
        out_ln = (inputs - ln_mean) / fluid.layers.sqrt(ln_var + self.eps)

        out = fluid.layers.expand(x=self.rho, expand_times=[inputs.shape[0], 1, inputs.shape[2], inputs.shape[3]]) * out_in + \
              fluid.layers.expand(x=self.rho, expand_times=[inputs.shape[0], 1, inputs.shape[2], inputs.shape[3]]) * out_ln
        out = out * fluid.layers.expand(x=self.gamma, expand_times=[inputs.shape[0], 1, inputs.shape[2], inputs.shape[3]]) + \
              fluid.layers.expand(x=self.beta, expand_times=[inputs.shape[0], 1, inputs.shape[2], inputs.shape[3]])
        return out

    def clip_rho(self, rho_clipper):
        self.rho.set_value(rho_clipper(self.rho))


class Discriminator(fluid.dygraph.Layer):
    def __init__(self, input_nc, ndf=64, n_layers=5):
        super(Discriminator, self).__init__()
        self.input_nc = input_nc
        self.ndf = ndf
        self.n_layers = n_layers

    def forward(self, inputs):
        x = fluid.layers.pad2d(input=inputs, paddings=[1,1,1,1], mode='reflect')
        x = SpectralNormConv2D(self.input_nc, self.ndf, 4, 2, 0, True)(x)
        x = fluid.layers.leaky_relu(x=x, alpha=0.2)

        for i in range(1, self.n_layers - 2):
            mult = 2 ** (i - 1)
            x = fluid.layers.pad2d(input=x, paddings=[1,1,1,1], mode='reflect')
            x = SpectralNormConv2D(self.ndf*mult, self.ndf*mult*2, filter_size=4, stride=2, padding=0, bias_attr=True)(x)
            x = fluid.layers.leaky_relu(x=x, alpha=0.2)

        mult = 2 ** (self.n_layers - 2 - 1)
        x = fluid.layers.pad2d(input=x, paddings=[1,1,1,1], mode='reflect')
        x = SpectralNormConv2D(self.ndf*mult, self.ndf*mult*2, filter_size=4, stride=1, padding=0, bias_attr=True)(x)
        x = fluid.layers.leaky_relu(x=x, alpha=0.2)

        mult = 2 ** (self.n_layers - 2)
        gap = fluid.layers.adaptive_pool2d(x, pool_size=1, pool_type='avg')
        self.gap_fc = SpectralNormLinear(self.ndf*mult, 1, bias_attr=False)
        gap_logit = self.gap_fc(fluid.layers.reshape(x=gap, shape=[x.shape[0], -1]))
        gap_weight = fluid.layers.transpose(list(self.gap_fc.parameters())[0], perm=[1,0])
        gap = x * fluid.layers.unsqueeze(input=gap_weight, axes=[2,3])

        gmp = fluid.layers.adaptive_pool2d(x, pool_size=1, pool_type='max')
        self.gmp_fc = SpectralNormLinear(self.ndf*mult, 1, bias_attr=False)
        gmp_logit = self.gmp_fc(fluid.layers.reshape(x=gmp, shape=[x.shape[0], -1]))
        gmp_weight = fluid.layers.transpose(list(self.gmp_fc.parameters())[0], perm=[1,0])
        gmp = x * fluid.layers.unsqueeze(input=gmp_weight, axes=[2,3])

        cam_logit = fluid.layers.concat([gap_logit, gmp_logit], 1)
        x = fluid.layers.concat([gap, gmp], 1)
        x = Conv2D(self.ndf * mult * 2, self.ndf * mult, filter_size=1, stride=1, bias_attr=True)(x)       
        x = fluid.layers.leaky_relu(x=x, alpha=0.2)

        heatmap = fluid.layers.reduce_sum(x, dim=1, keep_dim=True)
        
        x = fluid.layers.pad2d(input=x, paddings=[1,1,1,1], mode='reflect')
        out = SpectralNormConv2D(self.ndf * mult, 1, filter_size=4, stride=1, padding=0, bias_attr=True)(x)

        cam_logit = fluid.layers.sigmoid(x=cam_logit)
        out = fluid.layers.sigmoid(x=out)
        return out, cam_logit, heatmap


class RhoClipper(object):
    def __init__(self, min, max):
        self.clip_min = min
        self.clip_max = max
        assert min < max

    def __call__(self, weight):
        return fluid.layers.clamp(weight, self.clip_min, self.clip_max)
