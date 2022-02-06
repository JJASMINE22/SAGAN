# -*- coding: UTF-8 -*-
'''
@Project ：CNN_LSTM
@File    ：train_cifar10.py
@IDE     ：PyCharm 
@Author  ：XinYi Huang
'''
import numpy as np
from tensorflow.keras import activations
from tensorflow.python.keras.engine.input_spec import InputSpec
from tensorflow.python.keras.utils import conv_utils
from tensorflow.keras.layers import (Input,
                                     Layer,
                                     Dense,
                                     Conv2D,
                                     SeparableConv2D,
                                     Add)
from tensorflow.keras.regularizers import l2
from tensorflow.keras import initializers
import tensorflow as tf


class ConvSN2D(Conv2D):
    def __init__(self,
                 sn_initializer=initializers.RandomNormal(0,1),
                 **kwargs):
        super(ConvSN2D, self).__init__(**kwargs)
        self.sn_initiralizer = sn_initializer

    def get_config(self):
        config = super(ConvSN2D, self).get_config()
        config.update({
            'sn_initializer': self.sn_initiralizer
        })
        return config

    def build(self, input_shape):

        super(ConvSN2D, self).build(input_shape)

        self.sn = self.add_weight(shape=(1, self.filters),
                                  initializer=self.sn_initiralizer,
                                  name='sn',
                                  trainable=False)
        self.built = True

    def call(self, input):
        def _l2normalize(v, eps=1e-12):
            return v / (tf.reduce_sum(v ** 2) ** 0.5 + eps)

        def power_iteration(W, u):

            _u = u
            _v = _l2normalize(tf.matmul(_u, tf.transpose(W, perm=[1, 0])))
            _u = _l2normalize(tf.matmul(_v, W))

            return _u, _v

        W_shape = self.kernel.shape.as_list()
        W_reshaped = tf.reshape(self.kernel, [-1, W_shape[-1]])
        _u, _v = power_iteration(W_reshaped, self.sn)
        sigma = tf.matmul(_v, W_reshaped)
        sigma = tf.matmul(sigma, tf.transpose(_u, perm=[1, 0]))
        W_bar = W_reshaped / sigma
        with tf.control_dependencies([self.sn.assign(_u)]):
            W_bar = tf.reshape(W_bar, W_shape)

        output = tf.nn.conv2d(input, W_bar, self.strides, self.padding.upper(),
                              conv_utils.convert_data_format(self.data_format, ndim=4))
        if self.use_bias:
            output = tf.nn.bias_add(output, self.bias,
                                    conv_utils.convert_data_format(self.data_format, ndim=4))

        if self.activation is not None:
            return self.activation(output)

        return output


class ConvSN2DTranspose(Conv2D):
    '''
    If the parent class properties have been assigned real values,
    the superclass cannot assign values to these properties
    '''
    def __init__(self,
                 filters,
                 kernel_size,
                 strides=(1, 1),
                 padding='valid',
                 output_padding=None,
                 data_format=None,
                 dilation_rate=(1, 1),
                 activation=None,
                 use_bias=True,
                 kernel_initializer=initializers.GlorotUniform(),
                 bias_initializer=initializers.Zeros(),
                 kernel_regularizer=None,
                 bias_regularizer=None,
                 activity_regularizer=None,
                 kernel_constraint=None,
                 bias_constraint=None,
                 sn_initiralizer=initializers.RandomNormal(0, 1),
                 **kwargs):
        super(ConvSN2DTranspose, self).__init__(
            filters=filters,
            kernel_size=kernel_size,
            strides=strides,
            padding=padding,
            data_format=data_format,
            dilation_rate=dilation_rate,
            activation=activations.get(activation),
            use_bias=use_bias,
            kernel_initializer=kernel_initializer,
            bias_initializer=bias_initializer,
            kernel_regularizer=kernel_regularizer,
            bias_regularizer=bias_regularizer,
            activity_regularizer=activity_regularizer,
            kernel_constraint=kernel_constraint,
            bias_constraint=bias_constraint,
            **kwargs)
        self.sn_initiralizer = sn_initiralizer

        self.output_padding = output_padding
        if self.output_padding is not None:
            self.output_padding = conv_utils.normalize_tuple(
                self.output_padding, 2, 'output_padding', allow_zero=True)
            for stride, out_pad in zip(self.strides, self.output_padding):
                if out_pad >= stride:
                    raise ValueError('Strides must be greater than output padding. '
                                     f'Received strides={self.strides}, '
                                     f'output_padding={self.output_padding}.')

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

        kernel_shape = self.kernel_size + (self.filters, in_channels)

        self.kernel = self.add_weight(
            name='kernel',
            shape=kernel_shape,
            initializer=self.kernel_initializer,
            regularizer=self.kernel_regularizer,
            constraint=self.kernel_constraint,
            trainable=True,
            dtype=self.dtype)

        self.sn = self.add_weight(shape=(1, in_channels),
                                  initializer=self.sn_initiralizer,
                                  name='sn',
                                  trainable=False)

        if self.use_bias:
            self.bias = self.add_weight(
                name='bias',
                shape=(self.filters,),
                initializer=self.bias_initializer,
                regularizer=self.bias_regularizer,
                constraint=self.bias_constraint,
                trainable=True,
                dtype=self.dtype)
        else:
            self.bias = None
        self.built = True

    def call(self, inputs):

        inputs_shape = tf.shape(inputs)
        batch_size = inputs_shape[0]
        if self.data_format == 'channels_first':
            h_axis, w_axis = 2, 3
        else:
            h_axis, w_axis = 1, 2

        dims = inputs.shape.as_list()
        height = dims[h_axis]
        width = dims[w_axis]
        height = height if height is not None else inputs_shape[h_axis]
        width = width if width is not None else inputs_shape[w_axis]

        kernel_h, kernel_w = self.kernel_size
        stride_h, stride_w = self.strides
        dilation_rate_y, dilation_rate_x = self.dilation_rate
        assert np.logical_and(np.greater_equal(dilation_rate_y, 1), np.greater_equal(dilation_rate_x, 1))

        # 获取反卷积特征形状
        if self.padding == 'same':
            out_height = height*stride_h
            out_width = height*stride_w
        elif self.padding == 'valid':
            if np.logical_and(np.equal(dilation_rate_y, 1), np.equal(dilation_rate_x, 1)) or not self.dilation_rate:
                out_height = height*stride_h + kernel_h - 1
                out_width = width*stride_w + kernel_w - 1
            elif np.logical_or(np.greater(dilation_rate_y, 1), np.greater(dilation_rate_x, 1)):
                out_height = height*stride_h+(kernel_h-1)*dilation_rate_y
                out_width = width*stride_w+(kernel_w-1)*dilation_rate_x
        else:
            raise ValueError("padding must be in the set {valid, same}")

        if self.data_format == 'channels_first':
            output_shape_tensor = tf.cast([batch_size, self.filters, out_height, out_width], dtype=tf.int32)
        else:
            output_shape_tensor = tf.cast([batch_size, out_height, out_width, self.filters], dtype=tf.int32)

        def _l2normalize(v, eps=1e-12):
            return v / (tf.reduce_sum(v ** 2) ** 0.5 + eps)

        def power_iteration(W, u):

            _u = u
            _v = _l2normalize(tf.matmul(_u, tf.transpose(W, perm=[1, 0])))
            _u = _l2normalize(tf.matmul(_v, W))

            return _u, _v

        # 谱归一化操作
        W_shape = self.kernel.shape.as_list()
        W_reshaped = tf.reshape(self.kernel, [-1, W_shape[-1]])
        _u, _v = power_iteration(W_reshaped, self.sn)
        sigma = tf.matmul(_v, W_reshaped)
        sigma = tf.matmul(sigma, tf.transpose(_u, perm=[1, 0]))
        W_bar = W_reshaped / sigma
        with tf.control_dependencies([self.sn.assign(_u)]):
            W_bar = tf.reshape(W_bar, W_shape)

        outputs = tf.nn.conv2d_transpose(
            inputs,
            W_bar,
            output_shape_tensor,
            strides=self.strides,
            padding=self.padding.upper(),
            data_format=conv_utils.convert_data_format(self.data_format, ndim=4),
            dilations=self.dilation_rate)

        # if not tf.executing_eagerly():
        #     # Infer the static output shape:
        #     out_shape = self.compute_output_shape(inputs.shape)
        #     outputs.set_shape(out_shape)

        if self.use_bias:
            outputs = tf.nn.bias_add(
                outputs,
                self.bias,
                data_format=conv_utils.convert_data_format(self.data_format, ndim=4))

        if self.activation is not None:
            return self.activation(outputs)
        return outputs

    def compute_output_shape(self, input_shape):

        input_shape = tf.TensorShape(input_shape).as_list()
        batch_size = input_shape[0]
        if self.data_format == 'channels_first':
            h_axis, w_axis = 2, 3
        else:
            h_axis, w_axis = 1, 2

        height = input_shape[h_axis]
        width = input_shape[w_axis]
        height = height if height is not None else input_shape[h_axis]
        width = width if width is not None else input_shape[w_axis]

        kernel_h, kernel_w = self.kernel_size
        stride_h, stride_w = self.strides
        dilation_rate_y, dilation_rate_x = self.dilation_rate
        assert np.logical_and(np.greater_equal(dilation_rate_y, 1), np.greater_equal(dilation_rate_x, 1))

        if self.padding == 'same':
            out_height = height * stride_h
            out_width = height * stride_w
        elif self.padding == 'valid':
            if np.logical_and(np.equal(dilation_rate_y, 1), np.equal(dilation_rate_x, 1)) or not self.dilation_rate:
                out_height = height * stride_h + kernel_h - 1
                out_width = width * stride_w + kernel_w - 1
            elif np.logical_or(np.greater(dilation_rate_y, 1), np.greater(dilation_rate_x, 1)):
                out_height = height * stride_h + (kernel_h - 1) * dilation_rate_y
                out_width = width * stride_w + (kernel_w - 1) * dilation_rate_x
        else:
            raise ValueError("padding must be in the set {valid, same}")

        if self.data_format == 'channels_first':
            output_shape_tensor = [batch_size, self.filters, out_height, out_width]
        else:
            output_shape_tensor = [batch_size, out_height, out_width, self.filters]
        return output_shape_tensor

    def get_config(self):
        config = super(ConvSN2DTranspose, self).get_config()
        config['output_padding'] = self.output_padding
        config['sn_initiralizer'] = self.sn_initiralizer

        return config


class Squeeze(Layer):
    def __init__(self,
                 **kwargs):
        super(Squeeze, self).__init__(**kwargs)

    def call(self, input, *args, **kwargs):
        assert len(input.shape.as_list()) == 4

        return tf.squeeze(input, axis=[1, 2, 3])


class DenseSN(Dense):
    def __init__(self, **kwargs):
        super(DenseSN, self).__init__(**kwargs)

    def get_config(self):
        config = super(DenseSN, self).get_config()
        return config

    def build(self, input_shape):

        super(DenseSN, self).build(input_shape)

        self.sn = self.add_weight(shape=(1, self.units),
                                  initializer=initializers.RandomNormal(0, 1),
                                  name='sn',
                                  trainable=False)

        self.built = True

    def call(self, inputs):
        def _l2normalize(v, eps=1e-12):
            return v / (tf.reduce_sum(v ** 2) ** 0.5 + eps)

        def power_iteration(W, u):
            _u = u
            _v = _l2normalize(tf.matmul(_u, tf.transpose(W)))  # matrix transpose
            _u = _l2normalize(tf.matmul(_v, W))
            return _u, _v

        W_shape = self.kernel.shape.as_list()
        # Flatten the Tensor
        W_reshaped = tf.reshape(self.kernel, [-1, W_shape[-1]])
        _u, _v = power_iteration(W_reshaped, self.sn)
        # Calculate Sigma
        sigma = tf.matmul(_v, W_reshaped)
        sigma = tf.matmul(sigma, tf.transpose(_u))
        # normalize it
        W_bar = W_reshaped / sigma
        # reshape weight tensor
        with tf.control_dependencies([self.sn.assign(_u)]):
            W_bar = tf.reshape(W_bar, W_shape)
        output = tf.matmul(inputs, W_bar)
        if self.use_bias:
            output = tf.nn.bias_add(output, self.bias)
        if self.activation is not None:
            output = self.activation(output)
        return output


class NearestUpSampling2D(Layer):
    def __init__(self,
                 method='nearest',
                 scale_factor=2,
                 **kwargs):
        super(NearestUpSampling2D, self).__init__(**kwargs)
        assert method in ['bilinear', 'lanczos3', 'lanczos5', 'bicubic',
                          'gaussian', 'nearest', 'area', 'mitchellcubic']
        self.method = method
        self.scale_factor = scale_factor

    def get_config(self):

        config = super(NearestUpSampling2D, self).get_config()
        config.update({
            'method': self.method,
            'scale_factor': self.scale_factor,
        })

        return config

    def call(self, input, *args, **kwargs):

        input_shape = input.shape.as_list()
        assert len(input_shape) == 4

        h, w = input_shape[1:-1]

        new_size = [h * self.scale_factor, w * self.scale_factor]
        return tf.image.resize(images=input, size=new_size, method=self.method)
