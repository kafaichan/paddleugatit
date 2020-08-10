import paddle.fluid as fluid
from paddle.fluid.dygraph import Conv2D, Pool2D, BatchNorm, Linear, Dropout, to_variable
from paddle.fluid.initializer import ConstantInitializer
import numpy as np


def resnet_generator(name, inputs, input_nc, output_nc, ngf=64, n_blocks=6, img_size=256, light=False):
    assert(n_blocks >= 0)    
    x = fluid.layers.pad2d(input=inputs, paddings=[3,3,3,3], mode='reflect')
    x = fluid.layers.conv2d(input=x, num_filters=ngf, filter_size=7, stride=1, padding=0, groups=1, bias_attr=False)
    x = fluid.layers.instance_norm(input=x)
    x = fluid.layers.relu(x=x)

    n_downsampling = 2
    for i in range(n_downsampling):
        mult = 2**i
        x = fluid.layers.pad2d(input=x, paddings=[1,1,1,1], mode='reflect')
        x = fluid.layers.conv2d(input=x, num_filters=ngf*mult*2, filter_size=3, stride=2, padding=0, groups=1, bias_attr=False)
        x = fluid.layers.instance_norm(input=x)
        x = fluid.layers.relu(x=x)

    mult = 2**n_downsampling
    for i in range(n_blocks):
        x = resnet_block('res_{}'.format(i), x, ngf*mult, use_bias=False)

    gap = fluid.layers.adaptive_pool2d(input=x, pool_size=1, pool_type='avg')
    gap_weight = fluid.layers.create_parameter(shape=[ngf*mult, 1], dtype='float32', 
        attr=fluid.ParamAttr(name=name+"_gap_weight"))
    gap_logit = fluid.layers.mul(x=gap, y=gap_weight)
    gap = x * fluid.layers.unsqueeze(fluid.layers.transpose(gap_weight, perm=[1,0]), axes=[2,3])

    gmp = fluid.layers.adaptive_pool2d(input=x, pool_size=1, pool_type='max')
    gmp_weight = fluid.layers.create_parameter(shape=[ngf*mult, 1], dtype='float32',
        attr=fluid.ParamAttr(name=name+"_gmp_weight"))
    gmp_logit = fluid.layers.mul(x=gmp, y=gmp_weight)
    gmp = x * fluid.layers.unsqueeze(fluid.layers.transpose(gmp_weight, perm=[1,0]), axes=[2,3])

    cam_logit = fluid.layers.concat(input=[gap_logit, gmp_logit], axis=1)
    x = fluid.layers.concat(input=[gap, gmp], axis=1)
    x = fluid.layers.conv2d(input=x, num_filters=ngf*mult, filter_size=1, stride=1, groups=1, bias_attr=False)
    x = fluid.layers.relu(x=x)

    heatmap = fluid.layers.reduce_sum(x, dim=1, keep_dim=True)

    if light:
        x_ = fluid.layers.adaptive_pool2d(input=x, pool_size=1, pool_type='avg')
        x_ = fluid.layers.fc(input=x_, size=ngf*mult, bias_attr=False)
        x_ = fluid.layers.relu(x=x_)
        x_ = fluid.layers.fc(input=x_, size=ngf*mult, bias_attr=False)
        x_ = fluid.layers.relu(x=x_)
    else:
        x_ = fluid.layers.fc(input=x, size=ngf*mult, bias_attr=False)
        x_ = fluid.layers.relu(x=x_)
        x_ = fluid.layers.fc(input=x_, size=ngf*mult, bias_attr=False)
        x_ = fluid.layers.relu(x=x_)
    gamma = fluid.layers.fc(input=x_, size=ngf*mult, bias_attr=False)
    beta = fluid.layers.fc(input=x_, size=ngf*mult, bias_attr=False)

    for i in range(n_blocks):
        x = resnet_ada_iln_block("{}_ada_{}".format(name, i), x, ngf*mult, False, gamma, beta)

    for i in range(n_downsampling):
        mult = 2 ** (n_downsampling-i)
        x = fluid.layers.image_resize(input=x, scale=2, resample='NEAREST')
        x = fluid.layers.pad2d(input=x, paddings=[1,1,1,1], mode='reflect')
        x = fluid.layers.conv2d(input=x, num_filters=int(ngf*mult/2), filter_size=3, stride=1, padding=0, groups=1, bias_attr=False)
        x = iln("{}_iln_{}".format(name, i), x, int(ngf*mult/2), eps=1e-5)
        x = fluid.layers.relu(x=x)

    x = fluid.layers.pad2d(input=x, paddings=[3,3,3,3], mode='reflect')
    x = fluid.layers.conv2d(input=x, num_filters=output_nc, filter_size=7, stride=1, padding=0, groups=1, bias_attr=False)
    out = fluid.layers.tanh(x=x)

    cam_logit = fluid.layers.sigmoid(x=cam_logit)
    return out, cam_logit, heatmap


def resnet_block(name, inputs, dim, use_bias):
    x = fluid.layers.pad2d(input=inputs, paddings=[1,1,1,1], mode='reflect')
    x = fluid.layers.conv2d(input=x, num_filters=dim, filter_size=3, stride=1, padding=0, groups=1, bias_attr=use_bias)
    x = fluid.layers.instance_norm(input=x)
    x = fluid.layers.relu(x=x)
    x = fluid.layers.pad2d(input=x, paddings=[1,1,1,1], mode='reflect')
    x = fluid.layers.conv2d(input=x, num_filters=dim, filter_size=3, stride=1, padding=0, groups=1, bias_attr=use_bias)
    x = fluid.layers.instance_norm(input=x)
    out = inputs + x
    return out


def resnet_ada_iln_block(name, inputs, dim, use_bias, gamma, beta):
    out = fluid.layers.pad2d(input=inputs, paddings=[1,1,1,1], mode='reflect')
    out = fluid.layers.conv2d(input=out, num_filters=dim, filter_size=3, stride=1, padding=0, groups=1, bias_attr=use_bias)
    out = ada_iln("{}_ada1".format(name), out, dim, gamma, beta)
    out = fluid.layers.relu(x=out)
    out = fluid.layers.pad2d(input=out, paddings=[1,1,1,1], mode='reflect')
    out = fluid.layers.conv2d(input=out, num_filters=dim, filter_size=3, stride=1, padding=0, groups=1, bias_attr=use_bias)
    out = ada_iln("{}_ada2".format(name), out, dim, gamma, beta)
    return out + inputs


def ada_iln(name, inputs, num_features, gamma, beta, eps=1e-5):
    rho = fluid.layers.create_parameter(shape=[1, num_features, 1, 1], dtype='float32', 
        attr=fluid.ParamAttr(name=name+"_rho"),
        default_initializer=ConstantInitializer(0.9))
    in_mean = fluid.layers.reduce_mean(input=inputs, dim=[2,3], keep_dim=True)
    in_var = fluid.layers.assign(np.var(inputs.numpy(), axis=(2,3), keepdims=True))
    out_in = (inputs-in_mean) / fluid.layers.sqrt(in_var+eps)

    ln_mean = fluid.layers.reduce_mean(input=inputs, dim=[1,2,3], keep_dim=True)
    ln_var = fluid.layers.assign(np.var(inputs.numpy(), axis=(1,2,3), keepdims=True))
    out_ln = (inputs-ln_mean) / fluid.layers.sqrt(ln_var + eps)

    out = fluid.layers.expand(x=rho, expand_times=[inputs.shape[0], 1, inputs.shape[2], inputs.shape[3]]) * out_in + \
          fluid.layers.expand(x=(1-rho), expand_times=[inputs.shape[0], 1, inputs.shape[2], inputs.shape[3]]) * out_ln
    out = out * fluid.layers.unsqueeze(gamma, axes=[2,3]) + fluid.layers.unsqueeze(beta, axes=[2,3])
    return out


def iln(name, inputs, num_features, eps=1e-5):
    rho = fluid.layers.create_parameter(shape=[1, num_features, 1, 1], dtype='float32',
        attr=fluid.ParamAttr(name=name+"_rho"),
        default_initializer=ConstantInitializer(0.0))
    gamma = fluid.layers.create_parameter(shape=[1, num_features, 1, 1], dtype='float32',
        attr=fluid.ParamAttr(name=name+"_gamma"),
        default_initializer=ConstantInitializer(1.0))
    beta = fluid.layers.create_parameter(shape=[1, num_features, 1, 1], dtype='float32',
        attr=fluid.ParamAttr(name=name+"_beta"),
        default_initializer=ConstantInitializer(0.0))

    in_mean = fluid.layers.reduce_mean(input=inputs, dim=[2,3], keep_dim=True)
    in_var = fluid.layers.assign(np.var(inputs.numpy(), axis=(2,3), keepdims=True))
    out_in = (inputs - in_mean) / fluid.layers.sqrt(in_var + eps)

    ln_mean = fluid.layers.reduce_mean(input=inputs, dim=[1,2,3], keep_dim=True)
    ln_var = fluid.layers.assign(np.var(inputs.numpy(), axis=(1,2,3), keepdims=True))
    out_ln = (inputs - ln_mean) / fluid.layers.sqrt(ln_var + eps)

    out = fluid.layers.expand(x=rho, expand_times=[inputs.shape[0], 1, inputs.shape[2], inputs.shape[3]]) * out_in + \
          fluid.layers.expand(x=rho, expand_times=[inputs.shape[0], 1, inputs.shape[2], inputs.shape[3]]) * out_ln
    out = out * fluid.layers.expand(x=gamma, expand_times=[inputs.shape[0], 1, inputs.shape[2], inputs.shape[3]]) + \
          fluid.layers.expand(x=beta, expand_times=[inputs.shape[0], 1, inputs.shape[2], inputs.shape[3]])
    return out


def spectral_norm_conv2d(name, inputs, num_channels, num_filters, filter_size, stride, padding, bias_attr):
    weight = fluid.layers.create_parameter(shape=[num_filters, num_channels, filter_size, filter_size], dtype='float32',
        attr=fluid.ParamAttr(name=name+"_w"))
    spec_weight = fluid.layers.spectral_norm(weight=weight)
    if bias_attr:
        bias = fluid.layers.create_parameter(shape=[num_channels,1], dtype='float32', attr=fluid.ParamAttr(name=name+"_b"))
        spec_bias = fluid.layers.spectral_norm(weight=bias)        
        spec_bias = fluid.layers.squeeze(input=spec_bias, axes=[1])
        out = paddle.nn.functional.conv2d(input=inputs, weight=spec_weight, bias=spec_bias, padding=padding, stride=stride)
    else:
        out = paddle.nn.functional.conv2d(input=inputs, weight=spec_weight, bias=bias_attr, padding=padding, stride=stride)
    return out


def spectral_norm_linear(name, x, inputs, in_dim, out_dim):
    weight = fluid.layers.create_parameter(shape=[in_dim, out_dim], type='float32', attr=fluid.ParamAttr(name=name+"_w"))
    spec_weight = fluid.layers.spectral_norm(weight=weight)
    logit = fluid.layers.mul(x=inputs, y=spec_weight)
    out = x * fluid.layers.unsqueeze(input=fluid.layers.transpose(spec_weight, perm=[1,0]), axes=[2,3])
    return logit, out


def descrimninator(name, inputs, input_nc, ndf=64, n_layers=5):
    x = fluid.layers.pad2d(input=inputs, paddings=[1,1,1,1], mode='reflect')
    x = spectral_norm_conv2d("{}_speconv_1".format(name), x, input_nc, ndf, 4, 2, 0, True)
    x = fluid.layers.leaky_relu(x=x, alpha=0.2)

    for i in range(1, n_layers-2):
        mult = 2**(i-1)
        x = fluid.layers.pad2d(input=x, paddings=[1,1,1,1], mode='reflect')
        x = spectral_norm_conv2d("{}_loopspeconv_{}".format(name, i), x, ndf*mult, ndf*mult*2, 4, 2, 0, True)
        x = fluid.layers.leaky_relu(x=x, alpha=0.2)

    mult = 2 ** (n_layers-2-1)
    x = fluid.layers.pad2d(input=x, paddings=[1,1,1,1], mode='reflect')
    x = spectral_norm_conv2d("{}_speconv_2".format(name), x, ndf*mult, ndf*mult*2, 4, 1, 0, True)
    x = fluid.layers.leaky_relu(x=x, alpha=0.2)

    mult = 2 ** (n_layers-2)
    gap = fluid.layers.adaptive_pool2d(x, pool_size=1, pool_type='avg')
    gap_logit, gap = spectral_norm_linear("{}_speclin_gap".format(name), x, gap, ndf*mult, 1)

    gmp = fluid.layers.adaptive_pool2d(x, pool_size=1, pool_type='max')
    gmp_logit, gmp = spectral_norm_linear("{}_speclin_gmp".format(name), x, gmp, ndf*mult, 1)

    cam_logit = fluid.layers.concat([gap_logit, gmp_logit], 1)
    x = fluid.layers.concat([gap, gmp], 1)
    x = fluid.layers.conv2d(input=x, num_channels=ndf*mult*2, num_filters=ndf*mult, filter_size=1, stride=1, groups=1, bias_attr=True)
    x = fluid.layers.leaky_relu(x=x, alpha=0.2)

    heatmap = fluid.layers.reduce_sum(x, dim=1, keep_dim=True)
    x = fluid.layers.pad2d(input=x, paddings=[1,1,1,1], mode='reflect')
    out = spectral_norm_conv2d("{}_speconv_3".format(name), x, ndf*mult, 1, 4, 1, 0, False)

    cam_logit = fluid.layers.sigmoid(x=cam_logit)
    out = fluid.layers.sigmoid(x=out)
    return out, cam_logit, heatmap