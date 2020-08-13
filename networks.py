import paddle
import paddle.fluid as fluid
from paddle.fluid.dygraph import Conv2D, Pool2D, BatchNorm, Linear, Dropout, to_variable
from paddle.fluid.initializer import ConstantInitializer


def resnet_generator(name, inputs, input_nc, output_nc, ngf=64, n_blocks=6, img_size=256, light=False):
    assert(n_blocks >= 0)
    with fluid.unique_name.guard(name + "_"):
        x = fluid.layers.pad2d(input=inputs, paddings=[3,3,3,3], mode='reflect')
        x = fluid.layers.conv2d(input=x, num_filters=ngf, filter_size=7, stride=1, padding=0, groups=1, bias_attr=False, 
            param_attr=fluid.ParamAttr(initializer=fluid.initializer.NormalInitializer(loc=0.0, scale=0.02)))
        x = fluid.layers.instance_norm(input=x, param_attr=fluid.ParamAttr(initializer=fluid.initializer.NormalInitializer(loc=0.0, scale=0.02)))
        x = fluid.layers.relu(x=x)

        n_downsampling = 2
        for i in range(n_downsampling):
            mult = 2**i
            x = fluid.layers.pad2d(input=x, paddings=[1,1,1,1], mode='reflect')
            x = fluid.layers.conv2d(input=x, num_filters=ngf*mult*2, filter_size=3, stride=2, padding=0, groups=1, bias_attr=False,
                param_attr=fluid.ParamAttr(initializer=fluid.initializer.NormalInitializer(loc=0.0, scale=0.02)))
            x = fluid.layers.instance_norm(input=x, param_attr=fluid.ParamAttr(initializer=fluid.initializer.NormalInitializer(loc=0.0, scale=0.02)))
            x = fluid.layers.relu(x=x)

        mult = 2**n_downsampling
        for i in range(n_blocks):
            x = resnet_block('res_{}'.format(i), x, ngf*mult, use_bias=False)

        gap = fluid.layers.adaptive_pool2d(input=x, pool_size=1, pool_type='avg')
        gap_weight = fluid.layers.create_parameter(shape=[ngf*mult, 1], dtype='float32', 
            attr=fluid.ParamAttr(name=name+"_gap_weight", initializer=fluid.initializer.NormalInitializer(loc=0.0, scale=0.02)))
        gap_logit = fluid.layers.mul(x=gap, y=gap_weight)
        gap = x * fluid.layers.unsqueeze(fluid.layers.transpose(gap_weight, perm=[1,0]), axes=[2,3])

        gmp = fluid.layers.adaptive_pool2d(input=x, pool_size=1, pool_type='max')
        gmp_weight = fluid.layers.create_parameter(shape=[ngf*mult, 1], dtype='float32',
            attr=fluid.ParamAttr(name=name+"_gmp_weight", initializer=fluid.initializer.NormalInitializer(loc=0.0, scale=0.02)))
        gmp_logit = fluid.layers.mul(x=gmp, y=gmp_weight)
        gmp = x * fluid.layers.unsqueeze(fluid.layers.transpose(gmp_weight, perm=[1,0]), axes=[2,3])

        cam_logit = fluid.layers.concat(input=[gap_logit, gmp_logit], axis=1)
        x = fluid.layers.concat(input=[gap, gmp], axis=1)
        x = fluid.layers.conv2d(input=x, num_filters=ngf*mult, filter_size=1, stride=1, groups=1, bias_attr=True, 
            param_attr=fluid.ParamAttr(initializer=fluid.initializer.NormalInitializer(loc=0.0, scale=0.02)))
        x = fluid.layers.relu(x=x)

        heatmap = fluid.layers.reduce_sum(x, dim=1, keep_dim=True)

        if light:
            x_ = fluid.layers.adaptive_pool2d(input=x, pool_size=1, pool_type='avg')
            x_ = fluid.layers.fc(input=x_, size=ngf*mult, bias_attr=False, param_attr=fluid.ParamAttr(initializer=fluid.initializer.NormalInitializer(loc=0.0, scale=0.02)))
            x_ = fluid.layers.relu(x=x_)
            x_ = fluid.layers.fc(input=x_, size=ngf*mult, bias_attr=False, param_attr=fluid.ParamAttr(initializer=fluid.initializer.NormalInitializer(loc=0.0, scale=0.02)))
            x_ = fluid.layers.relu(x=x_)
        else:
            x_ = fluid.layers.fc(input=x, size=ngf*mult, bias_attr=False, param_attr=fluid.ParamAttr(initializer=fluid.initializer.NormalInitializer(loc=0.0, scale=0.02)))
            x_ = fluid.layers.relu(x=x_)
            x_ = fluid.layers.fc(input=x_, size=ngf*mult, bias_attr=False, param_attr=fluid.ParamAttr(initializer=fluid.initializer.NormalInitializer(loc=0.0, scale=0.02)))
            x_ = fluid.layers.relu(x=x_)
        gamma = fluid.layers.fc(input=x_, size=ngf*mult, bias_attr=False, param_attr=fluid.ParamAttr(initializer=fluid.initializer.NormalInitializer(loc=0.0, scale=0.02)))
        beta = fluid.layers.fc(input=x_, size=ngf*mult, bias_attr=False, param_attr=fluid.ParamAttr(initializer=fluid.initializer.NormalInitializer(loc=0.0, scale=0.02)))

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

    return out, cam_logit, heatmap


def resnet_block(name, inputs, dim, use_bias):
    x = fluid.layers.pad2d(input=inputs, paddings=[1,1,1,1], mode='reflect')
    x = fluid.layers.conv2d(input=x, num_filters=dim, filter_size=3, stride=1, padding=0, groups=1, bias_attr=use_bias,
        param_attr=fluid.ParamAttr(initializer=fluid.initializer.NormalInitializer(loc=0.0, scale=0.02)))
    x = fluid.layers.instance_norm(input=x, param_attr=fluid.ParamAttr(initializer=fluid.initializer.NormalInitializer(loc=0.0, scale=0.02)))
    x = fluid.layers.relu(x=x)
    x = fluid.layers.pad2d(input=x, paddings=[1,1,1,1], mode='reflect')
    x = fluid.layers.conv2d(input=x, num_filters=dim, filter_size=3, stride=1, padding=0, groups=1, bias_attr=use_bias, 
        param_attr=fluid.ParamAttr(initializer=fluid.initializer.NormalInitializer(loc=0.0, scale=0.02)))
    x = fluid.layers.instance_norm(input=x, param_attr=fluid.ParamAttr(initializer=fluid.initializer.NormalInitializer(loc=0.0, scale=0.02)))
    out = inputs + x
    return out


def resnet_ada_iln_block(name, inputs, dim, use_bias, gamma, beta):
    out = fluid.layers.pad2d(input=inputs, paddings=[1,1,1,1], mode='reflect')
    out = fluid.layers.conv2d(input=out, num_filters=dim, filter_size=3, stride=1, padding=0, groups=1, bias_attr=use_bias,
        param_attr=fluid.ParamAttr(initializer=fluid.initializer.NormalInitializer(loc=0.0, scale=0.02)))
    out = ada_iln("{}_ada1".format(name), out, dim, gamma, beta)
    out = fluid.layers.relu(x=out)
    out = fluid.layers.pad2d(input=out, paddings=[1,1,1,1], mode='reflect')
    out = fluid.layers.conv2d(input=out, num_filters=dim, filter_size=3, stride=1, padding=0, groups=1, bias_attr=use_bias,
        param_attr=fluid.ParamAttr(initializer=fluid.initializer.NormalInitializer(loc=0.0, scale=0.02)))
    out = ada_iln("{}_ada2".format(name), out, dim, gamma, beta)
    return out + inputs


def ada_iln(name, inputs, num_features, gamma, beta, eps=1e-5):
    rho = fluid.layers.create_parameter(shape=[1, num_features, 1, 1], dtype='float32', 
        attr=fluid.ParamAttr(name=name+"_rho", initializer=ConstantInitializer(0.9)))
    in_mean = fluid.layers.reduce_mean(input=inputs, dim=[2,3], keep_dim=True)
    in_var = variance(inputs=inputs, mean=in_mean, dim=[2,3], keep_dim=True)
    out_in = (inputs-in_mean) / fluid.layers.sqrt(in_var+eps)

    ln_mean = fluid.layers.reduce_mean(input=inputs, dim=[1,2,3], keep_dim=True)
    ln_var = variance(inputs=inputs, mean=ln_mean, dim=[1,2,3], keep_dim=True)
    out_ln = (inputs-ln_mean) / fluid.layers.sqrt(ln_var + eps)

    out = rho * out_in + (1-rho) * out_ln   
    out = out * fluid.layers.unsqueeze(gamma, axes=[2,3]) + fluid.layers.unsqueeze(beta, axes=[2,3])
    return out


def iln(name, inputs, num_features, eps=1e-5):
    rho = fluid.layers.create_parameter(shape=[1, num_features, 1, 1], dtype='float32',
        attr=fluid.ParamAttr(name=name+"_rho", initializer=ConstantInitializer(0.0)))
    gamma = fluid.layers.create_parameter(shape=[1, num_features, 1, 1], dtype='float32',
        attr=fluid.ParamAttr(name=name+"_gamma", initializer=ConstantInitializer(1.0)))
    beta = fluid.layers.create_parameter(shape=[1, num_features, 1, 1], dtype='float32',
        attr=fluid.ParamAttr(name=name+"_beta", initializer=ConstantInitializer(0.0)))

    in_mean = fluid.layers.reduce_mean(input=inputs, dim=[2,3], keep_dim=True)
    in_var = variance(inputs=inputs, mean=in_mean, dim=[2,3], keep_dim=True)
    out_in = (inputs - in_mean) / fluid.layers.sqrt(in_var + eps)

    ln_mean = fluid.layers.reduce_mean(input=inputs, dim=[1,2,3], keep_dim=True)
    ln_var = variance(inputs=inputs, mean=ln_mean, dim=[2,3], keep_dim=True)
    out_ln = (inputs - ln_mean) / fluid.layers.sqrt(ln_var + eps)

    out = rho * out_in + (1-rho) * out_ln
    out = out * gamma + beta
    return out


def spectral_norm_conv2d(name, inputs, num_channels, num_filters, filter_size, stride, padding, bias_attr):
    weight = fluid.layers.create_parameter(shape=[num_filters, num_channels, filter_size, filter_size], dtype='float32',
        attr=fluid.ParamAttr(name=name+"_w", initializer=fluid.initializer.NormalInitializer(loc=0.0, scale=0.02)))
    spec_weight = fluid.layers.spectral_norm(weight=weight)
    if bias_attr:
        bias = fluid.layers.create_parameter(shape=[num_filters,1], dtype='float32', 
            attr=fluid.ParamAttr(name=name+"_b", initializer=fluid.initializer.NormalInitializer(loc=0.0, scale=0.02)))
        spec_bias = fluid.layers.spectral_norm(weight=bias)        
        out = paddle.nn.functional.conv2d(input=inputs, weight=spec_weight, bias=spec_bias, padding=padding, stride=stride)
    else:
        out = paddle.nn.functional.conv2d(input=inputs, weight=spec_weight, padding=padding, stride=stride)
    return out


def spectral_norm_linear(name, x, inputs, in_dim, out_dim):
    weight = fluid.layers.create_parameter(shape=[in_dim, out_dim], dtype='float32', 
        attr=fluid.ParamAttr(name=name+"_w", initializer=fluid.initializer.NormalInitializer(loc=0.0, scale=0.02)))
    spec_weight = fluid.layers.spectral_norm(weight=weight)
    logit = fluid.layers.mul(x=inputs, y=spec_weight)
    out = x * fluid.layers.unsqueeze(input=fluid.layers.transpose(spec_weight, perm=[1,0]), axes=[2,3])
    return logit, out


def discriminator(name, inputs, input_nc, ndf=64, n_layers=5):
    with fluid.unique_name.guard(name + "_"):
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
        x = fluid.layers.conv2d(input=x, num_filters=ndf*mult, filter_size=1, stride=1, groups=1, bias_attr=True,
            param_attr=fluid.ParamAttr(initializer=fluid.initializer.NormalInitializer(loc=0.0, scale=0.02)))
        x = fluid.layers.leaky_relu(x=x, alpha=0.2)

        heatmap = fluid.layers.reduce_sum(x, dim=1, keep_dim=True)
        x = fluid.layers.pad2d(input=x, paddings=[1,1,1,1], mode='reflect')
        out = spectral_norm_conv2d("{}_speconv_3".format(name), x, ndf*mult, 1, 4, 1, 0, False)

    return out, cam_logit, heatmap


def l1loss(inputs, label):
    diff = fluid.layers.abs(inputs-label)
    return fluid.layers.mean(diff)


def bce_with_logit_loss(inputs, label, reduction='mean'):
    out = fluid.layers.sigmoid_cross_entropy_with_logits(inputs, label)
    if reduction == 'mean':
        return fluid.layers.reduce_mean(out)
    elif reduction == 'sum':
        return fluid.layers.reduce_sum(out)
    return out

def variance(inputs, mean, dim, keep_dim=True):
    if keep_dim:
        deviation = fluid.layers.pow(inputs-mean, 2.0)
        out = fluid.layers.reduce_mean(deviation, dim=dim, keep_dim=True)
    return out