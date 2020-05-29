
import tensorflow as tf
from tensorflow.keras.layers import Layer
from tensorflow.keras import initializers, regularizers, constraints
from tensorflow.keras.layers import Activation

## New Approach: (2 conv2d operations)
class LightConv2D(Layer):
    def __init__(self,
        filters,
        kernel_size,
        strides=(1, 1),
        padding='SAME',
        data_format=None,
        dilation_rate=(1, 1),
        activation=None,
        use_bias=True,
        kernel_initializer='glorot_uniform',
        bias_initializer='zeros',
        kernel_regularizer=tf.keras.regularizers.l2(1e-4),
        bias_regularizer=None,
        activity_regularizer=None,
        kernel_constraint=None,
        bias_constraint=None,
        k=8):
        super(LightConv2D, self).__init__()

        self.filters = filters
        self.kernel_size = kernel_size
        self.strides = strides
        self.padding = padding
        self.data_format = data_format
        self.dilation_rate = dilation_rate
        self.activation = activation
        self.use_bias = use_bias
        self.kernel_initializer = initializers.get(kernel_initializer)
        self.bias_initializer = initializers.get(bias_initializer)
        self.kernel_regularizer = regularizers.get(kernel_regularizer)
        self.bias_regularizer = regularizers.get(bias_regularizer)
        self.activity_regularizer = regularizers.get(activity_regularizer)
        self.kernel_constraint = constraints.get(kernel_constraint)
        self.bias_constraint = constraints.get(bias_constraint)
        self.k = k

    def _get_weight(self, shape, name=None, bias=False):
        if bias:
            W = self.add_weight(
                shape=shape,
                initializer=self.bias_regularizer,
                name=name,
                regularizer=self.bias_regularizer,
                constraint=self.bias_constraint)
        else:
            W = self.add_weight(
                shape=shape,
                name=name,
                initializer=self.kernel_initializer,
                regularizer=self.kernel_regularizer,
                constraint=self.kernel_constraint)

        return W

    def build(self, input_shape):
        k1 = self.kernel_size[0]
        k2 = self.kernel_size[1]
        c = input_shape[-1]
        f = self.filters
        k = self.k

        self.w1 = self._get_weight([k1, k2, c, k], name="w1")
        self.w2 = self._get_weight([k1, k2, k, f], name="w2")

        self.w1_shape = [k1, k2, c, k]
        self.w2_shape = [k1, k2, k, f]

        if self.use_bias:
            self.b1 = self._get_weight([k], name="b1", bias=True)
            self.b2 = self._get_weight([f], name="b2", bias=True)

    def call(self, x):
        w1 = self.w1
        x = tf.nn.conv2d(x, w1, [1, self.strides[0], self.strides[1], 1], self.padding)

        if self.use_bias == True:
            x = x + self.b1
        x = Activation("relu")(x)

        w2 = self.w2
        x = tf.nn.conv2d(x, w2, [1, self.strides[0], self.strides[1], 1], self.padding)

        if self.use_bias == True:
            x = x + self.b2

        if self.activation:
            x = Activation(self.activation)(x)

        return x

    def get_config(self):
        config = super(LightConv2D, self).get_config()

        config.update({'filters': self.filters})
        config.update({'kernel_size': self.kernel_size})
        config.update({'strides': self.strides})
        config.update({'padding': self.padding})
        config.update({'data_format': self.data_format})
        config.update({'dilation_rate': self.dilation_rate})
        config.update({'activation': self.activation})
        config.update({'use_bias': self.use_bias})
        config.update({'kernel_initializer': self.kernel_initializer})
        config.update({'bias_initializer': self.bias_initializer}),
        config.update({'kernel_regularizer': self.kernel_regularizer})
        config.update({'bias_regularizer': self.bias_regularizer })
        config.update({'activity_regularizer': self.activity_regularizer})
        config.update({'kernel_constraint': self.kernel_constraint})
        config.update({'bias_constraint': self.bias_constraint})
        config.update({'k': self.k})

        return config
