
import tensorflow as tf
from tensorflow.keras.layers import Layer
from tensorflow.keras import initializers, regularizers, constraints
from tensorflow.keras.layers import Activation

class LightDense(Layer):
    def __init__(self,
        units,
        activation=None,
        use_bias=True,
        kernel_initializer='glorot_uniform',
        bias_initializer='zeros',
        kernel_regularizer=None,
        bias_regularizer=None,
        activity_regularizer=None,
        kernel_constraint=None,
        bias_constraint=None,
        k=8):
        super(LightDense, self).__init__()
        self.units = units
        self.activation = activation
        self.use_bias = use_bias
        self.kernel_initializer = kernel_initializer
        self.bias_initializer = bias_initializer
        self.kernel_regularizer = kernel_regularizer
        self.bias_regularizer = bias_regularizer
        self.activity_regularizer = activity_regularizer
        self.kernel_constraint = kernel_constraint
        self.bias_constraint = bias_constraint
        self.k = k

    def build(self, input_shape):
        ## Shape
        w1_shape = [input_shape[-1], self.k]
        w2_shape = [self.k, self.units]

        ## Weights
        self.w1 = self.add_weight(
            shape=w1_shape,
            initializer=self.kernel_initializer,
            regularizer=self.kernel_regularizer,
            constraint=self.kernel_constraint,
            trainable=True,
            name="w1"
        )

        self.w2 = self.add_weight(
            shape=w2_shape,
            initializer=self.kernel_initializer,
            regularizer=self.kernel_regularizer,
            constraint=self.kernel_constraint,
            trainable=True,
            name="w2"
        )

        ## Bias
        if self.use_bias:
            bias_shape = [self.units]
            self.bias = self.add_weight(
                shape=bias_shape,
                initializer=self.bias_initializer,
                regularizer=self.bias_regularizer,
                constraint=self.bias_constraint,
                trainable=True,
                name="bias"
            )

    def call(self, x):
        # W = tf.matmul(self.w1, self.w2) * tf.sqrt(2/self.units)
        # y = tf.matmul(x, W)

        w1 = self.w1 * tf.sqrt(2/self.units)
        w2 = self.w2 * tf.sqrt(2/self.units)

        y = tf.matmul(x, w1)
        y = tf.matmul(y, w2)

        if self.use_bias:
            y = y + self.bias

        if self.activation:
            y = Activation(self.activation)(y)

        return y

    def get_config(self):
        config = super(LightDense, self).get_config()

        config.update({'units': self.units})
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
