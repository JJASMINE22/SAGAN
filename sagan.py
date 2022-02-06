# -*- coding: UTF-8 -*-
'''
@Project ：CNN_LSTM
@File    ：train_cifar10.py
@IDE     ：PyCharm
@Author  ：XinYi Huang
'''
import numpy as np
import tensorflow as tf
import skimage.io
from tensorflow.keras.layers import Input
from big_scale_residual_net import CreateModel

class SAGAN:
    def __init__(self,
                 learning_rate: list,
                 beta: list,
                 loss_mode:str,
                 **kwargs):
        """
        :param learning_rate: 生成器、对抗其学习率
        :param beta: 优化器beta参数
        :param loss_mode: 误差模式
        """
        assert np.logical_and(len(learning_rate)==2, len(beta)==2)
        assert loss_mode in ['binary', 'hinge', 'MSE', 'MAE']

        self.img_size = 128

        self.learning_rate = learning_rate
        self.beta = beta
        self.loss_mode = loss_mode

        self.generator = CreateModel().build_generator(Input(shape=(100,)))
        self.discriminator = CreateModel().build_discriminator(Input(shape=(self.img_size, self.img_size, 3)))

        self.G_optimizer = tf.keras.optimizers.Adam(learning_rate=self.learning_rate[0],
                                                    beta_1=self.beta[0], beta_2=self.beta[1])
        self.D_optimizer = tf.keras.optimizers.Adam(learning_rate=self.learning_rate[1],
                                                    beta_1=self.beta[0], beta_2=self.beta[1])

        self.train_G_loss = tf.keras.metrics.Mean()
        self.train_D_loss = tf.keras.metrics.Mean()
        self.train_G_accuracy = tf.keras.metrics.BinaryAccuracy(threshold=0.5)
        self.train_D_accuracy = tf.keras.metrics.BinaryAccuracy(threshold=0.5)

        self.test_G_loss = tf.keras.metrics.Mean()
        self.test_D_loss = tf.keras.metrics.Mean()
        self.test_G_accuracy = tf.keras.metrics.BinaryAccuracy(threshold=0.5)
        self.test_D_accuracy = tf.keras.metrics.BinaryAccuracy(threshold=0.5)

    def generator_loss(self, fake_label):

        if self.loss_mode == 'hinge':
            fake_loss = -tf.reduce_mean(fake_label)  # 增大loss距离
        if self.loss_mode == 'MSE':
            fake_loss = tf.reduce_mean(tf.square(tf.ones_like(fake_label)-fake_label))
        elif self.loss_mode == 'MAE':
            fake_loss = tf.reduce_mean(tf.abs(tf.ones_like(fake_label)-fake_label))
        else:
            fake_loss = tf.reduce_mean(tf.keras.losses.binary_crossentropy(tf.ones_like(fake_label), fake_label))

        return fake_loss

    def discriminator_loss(self, real_label, fake_label):

        if self.loss_mode == 'hinge':
            real_loss = tf.reduce_mean(tf.nn.relu(1.0 - real_label))  # abs(real-1), 正向
            fake_loss = tf.reduce_mean(tf.nn.relu(1.0 + fake_label))  # abs(fake-(-1)), 负向
        elif self.loss_mode == 'MSE':
            real_loss = tf.reduce_mean(tf.square(tf.zeros_like(fake_label)-fake_label))
            fake_loss = tf.reduce_mean(tf.square(tf.ones_like(real_label)-real_label))
        elif self.loss_mode == 'MAE':
            real_loss = tf.reduce_mean(tf.abs(tf.zeros_like(fake_label)-fake_label))
            fake_loss = tf.reduce_mean(tf.abs(tf.ones_like(real_label)-real_label))
        else:
            real_loss = tf.reduce_mean(tf.keras.losses.binary_crossentropy(tf.ones_like(real_label), real_label))
            fake_loss = tf.reduce_mean(tf.keras.losses.binary_crossentropy(tf.zeros_like(fake_label), fake_label))

        return real_loss + fake_loss

    @tf.function
    def train(self, noises, real_sources):
        """
        同步优化生成、对抗器,
        解开注释, 亦可异步优化,
        """
        with tf.GradientTape(persistent=True) as tape:
            fake_sources = self.generator(noises)
            fake_labels = self.discriminator(fake_sources)
            real_labels = self.discriminator(real_sources)
            G_loss = self.generator_loss(fake_labels)
            D_loss = self.discriminator_loss(real_labels, fake_labels)
        G_gradients = tape.gradient(G_loss, self.generator.trainable_variables)
        D_gradients = tape.gradient(D_loss, self.discriminator.trainable_variables)
        self.G_optimizer.apply_gradients(zip(G_gradients, self.generator.trainable_variables))
        self.D_optimizer.apply_gradients(zip(D_gradients, self.discriminator.trainable_variables))

        # with tf.GradientTape() as tape:
        #     fake_sources = self.generator(noises)
        #     fake_labels = self.discriminator(fake_sources)
        #     G_loss = self.generator_loss(fake_labels)
        # G_gradients = tape.gradient(G_loss, self.generator.trainable_variables)
        # self.G_optimizer.apply_gradients(zip(G_gradients, self.generator.trainable_variables))

        self.train_G_loss(G_loss)
        self.train_D_loss(D_loss)

        self.train_G_accuracy(tf.ones_like(fake_labels), fake_labels)
        self.train_D_accuracy(tf.concat([tf.ones_like(real_labels), tf.zeros_like(fake_labels)], axis=0),
                              tf.concat([real_labels, fake_labels], axis=0))

    @tf.function
    def test(self, noises, real_sources):

        with tf.GradientTape() as tape:
            fake_sources = self.generator(noises)
            fake_labels = self.discriminator(fake_sources)
            real_labels = self.discriminator(real_sources)
            G_loss = self.generator_loss(fake_labels)
            D_loss = self.discriminator_loss(real_labels, fake_labels)

        self.test_G_loss(G_loss)
        self.test_D_loss(D_loss)

        self.test_G_accuracy(tf.ones_like(fake_labels), fake_labels)
        self.test_D_accuracy(tf.concat([tf.ones_like(real_labels), tf.zeros_like(fake_labels)], axis=0),
                              tf.concat([real_labels, fake_labels], axis=0))

    def sample_images(self, epoch):

        count, r, c = 0, 1, 1

        noises = np.random.normal(0, 1, size=(r*c, 100))
        gen_images = self.generator(noises)
        images = (gen_images + 1) * 127.5

        sampled_images = np.zeros(shape=[r*self.img_size, c*self.img_size, 3])
        for i in range(r):
            for j in range(c):
                sampled_images[i*self.img_size:(i+1)*self.img_size, j*self.img_size:(j+1)*self.img_size, :] = images[count]
                count += 1

        skimage.io.imsave(".\\generate_images\\epoch_%d.png" % epoch, sampled_images.astype(np.uint8))
