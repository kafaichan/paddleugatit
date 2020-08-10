import paddle
import paddle.fluid as fluid
import paddle.fluid.dygraph as dygraph
from paddle.fluid.dygraph import Linear
from paddle.fluid.dygraph.base import to_variable

from paddle.fluid.initializer import XavierInitializer, NormalInitializer, ConstantInitializer
import numpy as np
import math
import paddle.fluid as fluid
from paddle.fluid.param_attr import ParamAttr

path = './infer_model'
infer = False

class MyLinear:
    def __init__(self):
        self.rho = fluid.layers.create_parameter(shape=[1], dtype='float32', attr=ParamAttr(name='rho', initializer=ConstantInitializer(1.0)), is_bias=False)

    def forward(self, x):
        out = fluid.layers.fc(input=x, size=1, act=None, param_attr=ParamAttr(name='myfc', initializer=ConstantInitializer(2.0)), bias_attr=False)
        out = out + self.rho
        return out


if infer:
    exe = fluid.Executor(fluid.CPUPlace())
    inputs = np.array([[1.0]]).astype('float32')
    [inference_program, feed_target_names, fetch_targets] = fluid.io.load_inference_model(dirname=path, executor=exe)
    results = exe.run(inference_program, feed={feed_target_names[0]: inputs}, fetch_list=fetch_targets)

    print(results)
else:
    train_data=np.array([[1.0],[2.0],[3.0],[4.0]]).astype('float32')
    y_true = np.array([[2.0],[4.0],[6.0],[8.0]]).astype('float32')

    x = fluid.layers.data(name='x', shape=[1], dtype='float32')
    y = fluid.layers.data(name='y', shape=[1], dtype='float32')

    t = MyLinear().forward(x)
    y_pred = MyLinear().forward(t)

    cost = fluid.layers.square_error_cost(input=y_pred, label=y)
    avg_cost = fluid.layers.mean(cost)

    sgd_optimizer =fluid.optimizer.SGD(learning_rate=0.01)
    sgd_optimizer.minimize(avg_cost)

    cpu = fluid.core.CPUPlace()
    exe = fluid.Executor(cpu)
    exe.run(fluid.default_startup_program())

    for i in range(100):
        outs = exe.run(feed={'x': train_data, 'y': y_true},
            fetch_list=[y_pred.name, t.name])
        #ret = fluid.global_scope().find_var('myfc').get_tensor()
        #print(np.array(ret))
        ret = fluid.global_scope().find_var('rho').get_tensor()
        print(outs)

    fluid.io.save_inference_model(dirname=path, feeded_var_names=['x'], target_vars=[y_pred], executor=exe)