# -*- coding: UTF-8 -*-
'''
@Project ：SAGAN
@File    ：big_scale_residual_net.py
@IDE     ：PyCharm
@Author  ：XinYi Huang
'''
import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import (Input,
                                     Flatten,
                                     Reshape,
                                     BatchNormalization,
                                     ReLU,
                                     LeakyReLU
                                     )
from tensorflow.keras.models import Model, Sequential
from scale_self_attention import ScaleSelfAttention, GoogleSelfAttention
from CustomLayers import (ConvSN2D,
                          ConvSN2DTranspose,
                          DenseSN,
                          NearestUpSampling2D)

class CreateModel(Model):
    """
    自注意力生成式对抗网络
    用于生成复杂且大尺度图像
    """
    def __init__(self,
                 g_chs=1024,
                 d_chs=64,
                 layer_nums=4,
                 **kwargs):
        super(CreateModel, self).__init__(**kwargs)
        self.g_chs = g_chs
        self.d_chs = d_chs
        self.layer_nums = layer_nums

    def resblock(self, init, out_channels, short_cut=False, is_D=False, down_sample=False):
        """
        :param init: 输入特征
        :param out_channels: 输出通道
        :param short_cut: 中途是否减少参量
        :param is_D: 是否为对抗器
        :param down_sample: 是否下采样
        :return: 输出特征
        全连接线性运算、卷积运算均使用谱归一化
        """

        x = ConvSN2D(filters=out_channels//2 if short_cut else out_channels,
                     kernel_size=3, strides=1, padding='SAME')(init)
        if is_D:
            x = LeakyReLU(alpha=0.2)(x)
        else:
            x = BatchNormalization()(x)
            x = ReLU()(x)

        x = ConvSN2D(filters=out_channels, kernel_size=3,
                     strides=2 if down_sample else 1,
                     padding='SAME')(x)

        if np.not_equal(init.shape[-1], out_channels) or down_sample:
            init = ConvSN2D(filters=out_channels, kernel_size=3,
                            strides=2 if down_sample else 1,
                            padding='SAME')(init)

        x = init + x
        if is_D:
            x = LeakyReLU(alpha=0.2)(x)
        else:
            x = BatchNormalization()(x)
            x = ReLU()(x)

        return x

    def init_resblock(self, init, out_channels, is_D=False, short_cut=False, down_sample=False):

        x = ConvSN2D(filters=out_channels // 2 if short_cut else out_channels,
                     kernel_size=3, strides=1, padding='SAME')(init)
        if is_D:
            x = LeakyReLU(alpha=0.2)(x)
        else:
            x = BatchNormalization()(x)
            x = ReLU()(x)

        x = ConvSN2D(filters=out_channels, kernel_size=3,
                     strides=2 if down_sample else 1, padding='SAME')(x)

        init = ConvSN2D(filters=out_channels, kernel_size=3,
                        strides=2 if down_sample else 1, padding='SAME')(init)

        x = init + x
        if is_D:
            x = LeakyReLU(alpha=0.2)(x)
        else:
            x = BatchNormalization()(x)
            x = ReLU()(x)

        return x

    def build_generator(self, input):
        """
        于4×4→128×128, 通道将变为2048,
        为控制参量, 第一个残差块使用同等通道卷积init_resblock
        """

        x = DenseSN(units=self.g_chs*16)(input)
        x = Reshape(target_shape=(4, 4, self.g_chs))(x)
        x = BatchNormalization()(x)
        x = ReLU()(x)

        x = NearestUpSampling2D(method='bicubic')(x)
        x = self.init_resblock(x, self.g_chs, short_cut=True)

        # x = ScaleSelfAttention(kernel_size=(3, 3), strides=(1, 1), padding='SAME',
        #                         data_format='channels_last', return_attention=False)(x)

        for i in range(2):
            x = NearestUpSampling2D(method='bicubic')(x)
            x = self.resblock(x, self.g_chs//2, short_cut=True)
            self.g_chs //= 2

        x = ScaleSelfAttention(kernel_size=(3, 3), strides=(1, 1), padding='SAME',
                                data_format='channels_last', return_attention=False)(x)

        for i in range(2):
            x = NearestUpSampling2D(method='bicubic')(x)
            x = self.resblock(x, self.g_chs//2, short_cut=True)
            self.g_chs //= 2

        output = ConvSN2D(filters=3, kernel_size=3, strides=1, padding='SAME',
                          activation='tanh')(x)

        return Model(input, output)

    def build_discriminator(self, input):
        """
        为防止对抗器过强, 导致过学习, 使得生成器陷入局部优化,
        控制参量, 于最后一个残差块使用同等通道卷积init_resblock
        若仍出现图像生成效果差的现象, 解开注释部分替代该结构
        """

        x = ConvSN2D(filters=self.d_chs, kernel_size=3, strides=1,
                     padding='SAME')(input)
        x = LeakyReLU(alpha=0.2)(x)

        for i in range(2):
            x = self.resblock(x, self.d_chs*2, short_cut=True, is_D=True, down_sample=True)
            self.d_chs *= 2

        x = ScaleSelfAttention(kernel_size=(3, 3), strides=(1, 1), padding='SAME',
                                data_format='channels_last', return_attention=False)(x)

        for i in range(3):
            if i < 2:
                x = self.resblock(x, self.d_chs*2, short_cut=True, is_D=True, down_sample=True)
                self.d_chs *= 2
            else:
                # x = ScaleSelfAttention(kernel_size=(3, 3), strides=(1, 1), padding='SAME',
                #                        data_format='channels_last', return_attention=False)(x)
                x = self.init_resblock(x, self.d_chs, is_D=True, short_cut=True, down_sample=True)

        x = Flatten()(x)
        output = DenseSN(units=1, activation='sigmoid')(x)

        # x = self.init_resblock(input, self.d_chs, is_D=True, short_cut=True, to_down=True)
        #
        # x = self.resblock(x, self.d_chs*2, is_D=True, short_cut=True, to_down=True)
        # 
        # x = ScaleSelfAttention(kernel_size=(1, 1), strides=(1, 1), padding='SAME',
        #                         data_format='channels_last', return_attention=False)(x)
        # for i in range(4):
        #     self.d_chs *= 2
        #     if i < 3:
        #         x = self.resblock(x, self.d_chs*2, is_D=True, short_cut=True, to_down=True)
        #     else:
        #         x = self.resblock(x, self.d_chs, is_D=True, short_cut=True, to_down=False)
        # x = Flatten()(x)
        # output = DenseSN(units=1, activation='sigmoid')(x)

        return Model(input, output)


if __name__ == '__main__':

    model = CreateModel()
    g_input = Input(shape=(100,))
    d_input = Input(shape=(128, 128, 3))
    generator = model.build_generator(g_input)
    discriminator = model.build_discriminator(d_input)
    generator.summary()
    discriminator.summary()
