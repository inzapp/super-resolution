"""
Authors : inzapp

Github url : https://github.com/inzapp/super-resolution

Copyright (c) 2023 Inzapp

Permission is hereby granted, free of charge, to any person obtaining
a copy of this software and associated documentation files (the
"Software"), to deal in the Software without restriction, including
without limitation the rights to use, copy, modify, merge, publish,
distribute, sublicense, and/or sell copies of the Software, and to
permit persons to whom the Software is furnished to do so, subject to
the following conditions:

The above copyright notice and this permission notice shall be
included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE
LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION
OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION
WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
"""
import os
import numpy as np
import tensorflow as tf

import cv2
from glob import glob
from time import time
from model import Model
from lr_scheduler import LRScheduler
from generator import DataGenerator


class SuperResolution:
    def __init__(self,
                 train_image_path,
                 validation_image_path,
                 input_shape=(32, 32, 1),
                 target_scale=2,
                 lr=1e-3,
                 batch_size=32,
                 save_interval=2000,
                 iterations=100000,
                 view_grid_size=4,
                 d_loss_ignore_threshold=0.01,
                 checkpoint_path='checkpoint',
                 use_gan=False,
                 training_view=False):
        assert input_shape[2] in [1, 3]
        self.input_shape = input_shape
        assert target_scale in [2, 4, 8, 16, 32]
        self.output_shape = (self.input_shape[0] * target_scale, self.input_shape[1] * target_scale, self.input_shape[2])
        self.lr = lr
        self.batch_size = batch_size
        self.save_interval = save_interval
        self.iterations = iterations
        self.view_grid_size = view_grid_size
        self.use_gan = use_gan
        self.training_view = training_view
        self.d_loss_ignore_threshold = d_loss_ignore_threshold
        self.checkpoint_path = checkpoint_path
        self.live_view_previous_time = time()

        self.model = Model(input_shape=input_shape, output_shape=self.output_shape, use_gan=use_gan)
        self.g_model, self.d_model, self.gan = self.model.build()

        self.train_image_paths = self.init_image_paths(train_image_path)
        self.validation_image_paths = self.init_image_paths(validation_image_path)
        self.train_data_generator = DataGenerator(
            generator=self.g_model,
            image_paths=self.train_image_paths,
            input_shape=input_shape,
            output_shape=self.output_shape,
            batch_size=batch_size)
        self.validation_data_generator = DataGenerator(
            generator=None,
            image_paths=self.validation_image_paths,
            input_shape=input_shape,
            output_shape=self.output_shape,
            batch_size=batch_size)

    def init_image_paths(self, image_path):
        return glob(f'{image_path}/**/*.jpg', recursive=True)

    def compute_gradient(self, model, optimizer, x, y_true, ignore_threshold=0.0):
        with tf.GradientTape() as tape:
            y_pred = model(x, training=True)
            loss = tf.reduce_mean(tf.square(y_true - y_pred))
            if loss < ignore_threshold:
                loss = 0.0
        gradients = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(gradients, model.trainable_variables))
        return loss

    def build_loss_str(self, iteration_count, d_loss, g_loss, gan_flag):
        loss_str = f'[iteration_count : {iteration_count:6d}]'
        if gan_flag:
            loss_str += f' d_loss: {d_loss:>8.4f}'
            loss_str += f', g_loss: {g_loss:>8.4f}'
        else:
            psnr = 20 * np.log10(1.0 / np.sqrt(g_loss)) if g_loss != 0.0 else 100.0
            loss_str += f' PSNR: {psnr:>8.2f}'
        return loss_str

    def fit(self):
        self.model.summary()
        print(f'\ntrain on {len(self.train_image_paths)} samples.')
        print('start training')
        gan_flag = False
        iteration_count = 0
        os.makedirs(self.checkpoint_path, exist_ok=True)
        if self.use_gan:
            g_optimizer = tf.keras.optimizers.RMSprop(lr=self.lr * 0.5)
            d_optimizer = tf.keras.optimizers.RMSprop(lr=self.lr * 0.1)
        m_optimizer = tf.keras.optimizers.RMSprop(lr=self.lr)
        if self.use_gan:
            compute_gradient_d = tf.function(self.compute_gradient)
            compute_gradient_g = tf.function(self.compute_gradient)
        compute_gradient_m = tf.function(self.compute_gradient)
        if self.use_gan:
            g_lr_scheduler = LRScheduler(lr=self.lr * 0.5, iterations=self.iterations, warm_up=0.1, policy='step')
            d_lr_scheduler = LRScheduler(lr=self.lr * 0.1, iterations=self.iterations, warm_up=0.1, policy='step')
        m_lr_scheduler = LRScheduler(lr=self.lr, iterations=self.iterations, warm_up=0.1, policy='step')
        while True:
            dx, dy, gx, gy = self.train_data_generator.load(gan_flag)
            if gan_flag:
                g_lr_scheduler.update(g_optimizer, iteration_count)
                d_lr_scheduler.update(d_optimizer, iteration_count)
                self.d_model.trainable = True
                d_loss = compute_gradient_d(self.d_model, d_optimizer, dx, dy, self.d_loss_ignore_threshold)
                self.d_model.trainable = False
                g_loss = compute_gradient_g(self.gan, g_optimizer, gx, gy)
            else:
                m_lr_scheduler.update(m_optimizer, iteration_count)
                d_loss = 0.0
                g_loss = compute_gradient_m(self.g_model, m_optimizer, gx, gy)
            iteration_count += 1
            print(self.build_loss_str(iteration_count, d_loss, g_loss, gan_flag))
            if self.use_gan:
                gan_flag = not gan_flag
            if self.training_view:
                self.training_view_function()
            if iteration_count % self.save_interval == 0:
                model_path_without_extention = f'{self.checkpoint_path}/generator_{iteration_count}_iter'
                self.g_model.save(f'{model_path_without_extention}.h5', include_optimizer=False)
                generated_images = self.generate_image_grid(grid_size=4)
                cv2.imwrite(f'{model_path_without_extention}.jpg', generated_images)
                print(f'[iteration count : {iteration_count:6d}] model with generated images saved with {model_path_without_extention} h5 and jpg\n')
            if iteration_count == self.iterations:
                print('\n\ntrain end successfully')
                self.show_sr_images()
                exit(0)

    @staticmethod
    @tf.function
    def graph_forward(model, x):
        return model(x, training=False)

    def sample_images(self, size):
        data_generator = self.validation_data_generator
        raw_images = np.asarray(data_generator.load_images(count=size, shape=self.output_shape, interpolation='auto', normalize=False)).astype('uint8')
        input_images_reduced = np.asarray([data_generator.resize(img, (self.input_shape[1], self.input_shape[0]), 'area') for img in raw_images]).astype('uint8')
        input_images_nearest = np.asarray([data_generator.resize(img, (self.output_shape[1], self.output_shape[0]), 'nearest') for img in input_images_reduced]).astype('uint8')
        input_images_bicubic = np.asarray([data_generator.resize(img, (self.output_shape[1], self.output_shape[0]), 'bicubic') for img in input_images_reduced]).astype('uint8')
        z = data_generator.normalize(input_images_reduced)
        y = np.asarray(self.graph_forward(self.g_model, z))
        sr_images = data_generator.denormalize(y).reshape((size,) + self.output_shape)
        
        target_shape = (size,) + self.output_shape[:2]
        if self.input_shape[-1] == 3:
            target_shape += (self.output_shape[-1],)

        raw_images = np.reshape(raw_images, target_shape)
        input_images_nearest = np.reshape(input_images_nearest, target_shape)
        input_images_bicubic = np.reshape(input_images_bicubic, target_shape)
        sr_images = np.reshape(sr_images, target_shape)
        return raw_images, input_images_nearest, input_images_bicubic, sr_images

    def make_border(self, img, size=5):
        return cv2.copyMakeBorder(img, size, size, size, size, None, value=(192, 192, 192)) 

    def training_view_function(self):
        cur_time = time()
        if cur_time - self.live_view_previous_time > 3.0:
            generated_images = self.generate_image_grid(grid_size=self.view_grid_size)
            cv2.imshow('sr_images', generated_images)
            cv2.waitKey(1)
            self.live_view_previous_time = cur_time

    def generate_image_grid(self, grid_size):
        raw_images, input_images_nearest, input_images_bicubic, sr_images = self.sample_images(size=grid_size)
        generated_image_grid = None
        for i in range(grid_size):
            raw_image_border = self.make_border(raw_images[i])
            input_image_nearest_border = self.make_border(input_images_nearest[i])
            input_image_bicubic_border = self.make_border(input_images_bicubic[i])
            sr_image_border = self.make_border(sr_images[i])
            grid_row = np.concatenate((raw_image_border, input_image_nearest_border, input_image_bicubic_border, sr_image_border), axis=1)
            if generated_image_grid is None:
                generated_image_grid = grid_row
            else:
                generated_image_grid = np.append(generated_image_grid, grid_row, axis=0)
        return generated_image_grid

    def show_sr_images(self):
        while True:
            generated_images = self.generate_image_grid(grid_size=self.view_grid_size)
            cv2.imshow('sr_images', generated_images)
            key = cv2.waitKey(0)
            if key == 27:
                break

