# -*- coding: UTF-8 -*-
'''
@Project ：CNN_LSTM
@File    ：train.py
@IDE     ：PyCharm 
@Author  ：XinYi Huang
'''
import os
import cv2
import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds
from sagan import SAGAN
from warmup_cosine_decay_scheduler import WarmUpCosineDecayScheduler

if __name__ == '__main__':
    Epochs = 500
    trainset_size = 3000
    testset_size = 1000
    batch_size = 32
    cosine_scheduler = False
    ckpt_path = ".\\car_models\\checkpoint"

    model = SAGAN(learning_rate=[1e-4, 3e-4],
                  beta=[0., 0.9],
                  loss_mode='binary')

    if not os.path.exists(ckpt_path):
        os.makedirs(ckpt_path)

    ckpt = tf.train.Checkpoint(generator=model.generator,
                               discriminator=model.discriminator,
                               G_optimizer=model.G_optimizer,
                               D_optimizer=model.D_optimizer)

    ckpt_manager = tf.train.CheckpointManager(ckpt, ckpt_path, max_to_keep=5)

    # 如果检查点存在，则恢复最新的检查点。
    if ckpt_manager.latest_checkpoint:
        ckpt.restore(ckpt_manager.latest_checkpoint)
        print('Latest checkpoint restored!!')

    if cosine_scheduler:
        total_steps = trainset_size//batch_size * Epochs
        warmup_steps = int(total_steps * 0.2)
        hold_steps = trainset_size//batch_size * batch_size
        G_reduce_lr = WarmUpCosineDecayScheduler(global_interval_steps=total_steps,
                                                 warmup_interval_steps=warmup_steps,
                                                 hold_interval_steps=hold_steps,
                                                 learning_rate_base=1e-4,
                                                 warmup_learning_rate=1e-5,
                                                 min_learning_rate=1e-7,
                                                 verbose=0)
        D_reduce_lr = WarmUpCosineDecayScheduler(global_interval_steps=total_steps,
                                                 warmup_interval_steps=warmup_steps,
                                                 hold_interval_steps=hold_steps,
                                                 learning_rate_base=5e-4,
                                                 warmup_learning_rate=5e-5,
                                                 min_learning_rate=5e-7,
                                                 verbose=0)

    # 使用cars196数据训练模型
    examples, metadata = tfds.load('cars196', with_info=True)
    train_sources, test_sources = examples['train'].take(trainset_size).map(lambda x: x['image']),\
                                  examples['test'].take(testset_size).map(lambda x: x['image'])

    train_sources_batch = []
    test_sources_batch = []
    for epoch in range(Epochs):

        for i, source in enumerate(train_sources):
            train_sources_batch.append(cv2.resize(np.array(source), (128, 128), interpolation=cv2.INTER_CUBIC))
            if np.equal(len(train_sources_batch), batch_size) or np.equal(i, len(train_sources)-1):
                print(i)
                sources = np.array(train_sources_batch.copy())
                train_sources_batch.clear()
                if cosine_scheduler:
                    G_learning_rate = G_reduce_lr.batch_begin()
                    D_learning_rate = D_reduce_lr.batch_begin()
                    model.G_optimizer.learning_rate = G_learning_rate
                    model.D_optimizer.learning_rate = D_learning_rate
                noises = np.random.normal(0, 1, size=(batch_size, 100))
                model.train(noises, sources.astype('float')/127.5-1.)

        model.sample_images(epoch)

        for i, source in enumerate(test_sources):
            test_sources_batch.append(cv2.resize(np.array(source), (128, 128), interpolation=cv2.INTER_CUBIC))
            if np.equal(len(test_sources_batch), batch_size) or np.equal(i, len(test_sources) - 1):
                sources = np.array(test_sources_batch.copy())
                test_sources_batch.clear()
                noises = np.random.normal(0, 1, size=(batch_size, 100))
                model.test(noises, sources.astype('float')/127.5-1.)

        print(
            f'Epoch {epoch + 1}, '
            f'G_Loss: {model.train_G_loss.result()}, '
            f'D_Loss: {model.train_D_loss.result()}, '
            f'G_acc: {model.train_G_accuracy.result()}, '
            f'D_acc: {model.train_D_accuracy.result()}, '
            f'Test G_Loss: {model.test_G_loss.result()}, '
            f'Test D_Loss: {model.test_D_loss.result()}, '
            f'Test G_acc: {model.test_G_accuracy.result()}, '
            f'Test D_acc: {model.test_D_accuracy.result()}, '
        )

        ckpt_save_path = ckpt_manager.save()

        # 在下一次迭代开始前, 重置记录
        model.train_G_loss.reset_states()
        model.train_D_loss.reset_states()
        model.train_G_accuracy.reset_states()
        model.train_D_accuracy.reset_states()
        model.test_G_loss.reset_states()
        model.test_D_loss.reset_states()
        model.test_G_accuracy.reset_states()
        model.test_D_accuracy.reset_states()
