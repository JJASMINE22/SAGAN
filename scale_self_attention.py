# -*- coding: UTF-8 -*-
'''
@Project ：CNN_LSTM
@File    ：train_cifar10.py
@IDE     ：PyCharm 
@Author  ：XinYi Huang
'''
import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Layer
from tensorflow.keras import initializers, regularizers, activations
from tensorflow.python.keras.utils import conv_utils


class GoogleSelfAttention(Layer):
    def __init__(self,
                 kernel_size: tuple,
                 strides: tuple,
                 padding: str,
                 data_format: str,
                 return_attention: bool,
                 kernel_initializer=initializers.GlorotUniform(),
                 kernel_regularizer=None,
                 kernel_constraint=None,
                 gamma_initializer=initializers.Zeros(),
                 gamma_regularizer=None,
                 gamma_constraint=None,
                 **kwargs):
        super(GoogleSelfAttention, self).__init__(**kwargs)
        self.kernel_size = kernel_size
        self.strides = strides
        self.padding = padding
        self.data_format = data_format
        self.return_attention = return_attention
        self.kernel_initializer = kernel_initializer
        self.kernel_regularizer = kernel_regularizer
        self.kernel_constraint = kernel_constraint
        self.gamma_initializer = gamma_initializer
        self.gamma_regularizer = gamma_regularizer
        self.gamma_constraint = gamma_constraint

    def get_config(self):
        config = super(GoogleSelfAttention, self).get_config()
        config.update({
            'kernel_size': self.kernel_size,
            'strides': self.strides,
            'padding': self.padding,
            'data_format': self.data_format,
            'return_attention': self.return_attention,
            'kernel_initializer': self.kernel_initializer,
            'kernel_regularizer': self.kernel_regularizer,
            'kernel_constraint': self.kernel_constraint,
            'gamma_initializer': self.gamma_initializer,
            'gamma_regularizer': self.gamma_regularizer,
            'gamma_constraint': self.gamma_constraint
        })

        return config

    def build(self, input_shape):
        assert len(input_shape) == 4

        if self.data_format == 'channels_first':
            channel_axis = 1
        else:
            channel_axis = -1

        in_channels = input_shape[channel_axis]

        if not in_channels:
            raise ValueError('The channel dimension of the inputs '
                             'should be defined. Found `None`.')

        kernel_f_shape = self.kernel_size + (in_channels, in_channels // 8)
        kernel_g_shape = self.kernel_size + (in_channels, in_channels // 8)
        kernel_h_shape = self.kernel_size + (in_channels, in_channels // 2)
        kernel_o_shape = self.kernel_size + (in_channels // 2, in_channels)

        self.kernel_f = self.add_weight(shape=kernel_f_shape,
                                        initializer=self.kernel_initializer,
                                        regularizer=self.kernel_regularizer,
                                        constraint=self.kernel_constraint,
                                        trainable=True,
                                        name='kernel_f')

        self.kernel_g = self.add_weight(shape=kernel_g_shape,
                                        initializer=self.kernel_initializer,
                                        regularizer=self.kernel_regularizer,
                                        constraint=self.kernel_constraint,
                                        trainable=True,
                                        name='kernel_g')

        self.kernel_h = self.add_weight(shape=kernel_h_shape,
                                        initializer=self.kernel_initializer,
                                        regularizer=self.kernel_regularizer,
                                        constraint=self.kernel_constraint,
                                        trainable=True,
                                        name='kernel_h')

        self.kernel_o = self.add_weight(shape=kernel_o_shape,
                                        initializer=self.kernel_initializer,
                                        regularizer=self.kernel_regularizer,
                                        constraint=self.kernel_constraint,
                                        trainable=True,
                                        name='kernel_o')

        self.gamma = self.add_weight(shape=(),
                                     initializer=self.gamma_initializer,
                                     regularizer=self.gamma_regularizer,
                                     constraint=self.gamma_constraint,
                                     trainable=True,
                                     name='gamma')

        self.built = True

    def hw_flatten(self, x):
        if self.data_format == 'channels_first':
            h, w = x.shape[2], x.shape[3]
            return tf.reshape(x, [tf.shape(x)[0], -1, h*w])
        else:
            h, w = x.shape[1], x.shape[2]
            return tf.reshape(x, [tf.shape(x)[0], h*w, -1])

    def transpose_matmul(self, x, y):
        if self.data_format == 'channels_first':
            return tf.matmul(tf.transpose(self.hw_flatten(x), [0, 2, 1]), self.hw_flatten(y))
        else:
            return tf.matmul(self.hw_flatten(x), tf.transpose(self.hw_flatten(y), [0, 2, 1]))

    def call(self, input, *args, **kwargs):

        bs = tf.shape(input)[0]
        if self.data_format == 'channels_first':
            height, width = input.shape[2], input.shape[3]
        else:
            height, width = input.shape[1], input.shape[2]

        f = tf.nn.conv2d(input, self.kernel_f, self.strides, self.padding,
                         conv_utils.convert_data_format(self.data_format, ndim=4))
        f = tf.nn.max_pool2d(f, ksize=2, strides=2, padding=self.padding,
                             data_format=conv_utils.convert_data_format(self.data_format, ndim=4))
        g = tf.nn.conv2d(input, self.kernel_g, self.strides, self.padding,
                         conv_utils.convert_data_format(self.data_format, ndim=4))
        h = tf.nn.conv2d(input, self.kernel_h, self.strides, self.padding,
                         conv_utils.convert_data_format(self.data_format, ndim=4))
        h = tf.nn.max_pool2d(h, ksize=2, strides=2, padding=self.padding,
                             data_format=conv_utils.convert_data_format(self.data_format, ndim=4))

        e = self.transpose_matmul(g, f)
        attention_map = tf.nn.softmax(e)

        if self.data_format == 'channels_first':
            o = tf.reshape(tf.matmul(self.hw_flatten(h), attention_map, transpose_b=True),
                           [bs, -1, height, width])
        else:
            o = tf.reshape(tf.matmul(attention_map, self.hw_flatten(h)),
                           [bs, height, width, -1])

        o = tf.nn.conv2d(o, self.kernel_o, self.strides, self.padding,
                         conv_utils.convert_data_format(self.data_format, ndim=4))

        x = tf.add(input, o * self.gamma)

        if self.return_attention:
            return x, attention_map
        return x


class ScaleSelfAttention(Layer):
    def __init__(self,
                 kernel_size: tuple,
                 strides: tuple,
                 padding: str,
                 data_format: str,
                 return_attention: bool,
                 kernel_initializer=initializers.GlorotUniform(),
                 kernel_constraint=None,
                 gamma_initializer=initializers.Zeros(),
                 gamma_constraint=None,
                 **kwargs):
        super(ScaleSelfAttention, self).__init__(**kwargs)
        self.kernel_size = kernel_size
        self.strides = strides
        self.padding = padding
        self.data_format = data_format
        self.return_attention = return_attention
        self.kernel_initializer = kernel_initializer
        self.kernel_constraint = kernel_constraint
        self.gamma_initializer = gamma_initializer
        self.gamma_constraint = gamma_constraint

    def get_config(self):

        config = super(ScaleSelfAttention, self).get_config()
        config.update({
            'kernel_size': self.kernel_size,
            'strides': self.strides,
            'padding': self.padding,
            'data_format': self.data_format,
            'return_attention': self.return_attention,
            'kernel_initializer': self.kernel_initializer,
            'kernel_constraint': self.kernel_constraint,
            'gamma_initializer': self.gamma_initializer,
            'gamma_constraint': self.gamma_constraint
        })

        return config

    def build(self, input_shape):

        assert len(input_shape) == 4

        if self.data_format == 'channels_first':
            channel_axis = 1
        else:
            channel_axis = -1

        in_channels = input_shape[channel_axis]

        if not in_channels:
            raise ValueError('The channel dimension of the inputs '
                             'should be defined. Found `None`.')

        kernel_f_shape = self.kernel_size + (in_channels, in_channels//8)
        kernel_g_shape = self.kernel_size + (in_channels, in_channels//8)
        kernel_h_shape = self.kernel_size + (in_channels,)*2

        self.kernel_f = self.add_weight(shape=kernel_f_shape,
                                        initializer=self.kernel_initializer,
                                        constraint=self.kernel_constraint,
                                        trainable=True,
                                        name='kernel_f')

        self.kernel_g = self.add_weight(shape=kernel_g_shape,
                                        initializer=self.kernel_initializer,
                                        constraint=self.kernel_constraint,
                                        trainable=True,
                                        name='kernel_g')

        self.kernel_h = self.add_weight(shape=kernel_h_shape,
                                        initializer=self.kernel_initializer,
                                        constraint=self.kernel_constraint,
                                        trainable=True,
                                        name='kernel_h')

        self.gamma = self.add_weight(shape=(),
                                     initializer=self.gamma_initializer,
                                     constraint=self.gamma_constraint,
                                     trainable=True,
                                     name='gamma')

        self.built = True

    def hw_flatten(self, x):
        if self.data_format == 'channels_first':
            h, w = x.shape[2], x.shape[3]
            return tf.reshape(x, [tf.shape(x)[0], -1, h*w])
        else:
            h, w = x.shape[1], x.shape[2]
            return tf.reshape(x, [tf.shape(x)[0], h*w, -1])

    def transpose_matmul(self, x, y):
        if self.data_format == 'channels_first':
            return tf.matmul(tf.transpose(self.hw_flatten(x), [0, 2, 1]), self.hw_flatten(y))
        else:
            return tf.matmul(self.hw_flatten(x), tf.transpose(self.hw_flatten(y), [0, 2, 1]))

    def call(self, input, *args, **kwargs):

        bs = tf.shape(input)[0]
        if self.data_format == 'channels_first':
            height, width = input.shape[2], input.shape[3]
        else:
            height, width = input.shape[1], input.shape[2]

        f = tf.nn.conv2d(input, self.kernel_f, self.strides, self.padding,
                         conv_utils.convert_data_format(self.data_format, ndim=4))
        g = tf.nn.conv2d(input, self.kernel_g, self.strides, self.padding,
                         conv_utils.convert_data_format(self.data_format, ndim=4))
        h = tf.nn.conv2d(input, self.kernel_h, self.strides, self.padding,
                         conv_utils.convert_data_format(self.data_format, ndim=4))

        e = self.transpose_matmul(f, g)
        attention_map = tf.nn.softmax(e)

        if self.data_format == 'channels_first':
            o = tf.reshape(tf.matmul(self.hw_flatten(h), attention_map, transpose_b=True),
                           [bs, -1, height, width])
        else:
            o = tf.reshape(tf.matmul(attention_map, self.hw_flatten(h)),
                           [bs, height, width, -1])

        x = tf.add(input, o * self.gamma)

        if self.return_attention:
            return x, attention_map
        return x
