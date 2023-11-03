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
import tensorflow as tf


os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'


class Model:
    def __init__(self, input_shape, output_shape, use_gan):
        self.input_shape = input_shape
        self.output_shape = output_shape
        self.target_scale = output_shape[0] / input_shape[0]
        self.use_gan = use_gan
        self.gan = None
        self.g_model = None
        self.d_model = None

    def build(self):
        assert self.output_shape[0] % self.target_scale == 0 and self.output_shape[1] % self.target_scale == 0
        g_input, g_output = self.build_g(bn=self.use_gan)
        self.g_model = tf.keras.models.Model(g_input, g_output)
        if self.use_gan:
            d_input, d_output = self.build_d(bn=False)
            self.d_model = tf.keras.models.Model(d_input, d_output)
            gan_output = self.d_model(g_output)
            self.gan = tf.keras.models.Model(g_input, gan_output)
        return self.g_model, self.d_model, self.gan

    # def build_g(self, bn):
    #     g_input = tf.keras.layers.Input(shape=self.input_shape)
    #     x = g_input
    #     end_filters = 16
    #     initial_filters = end_filters * (self.target_scale // 2)
    #     if self.target_scale >= 2:
    #         x = self.upsampling(x)
    #         x = self.conv2d(x, initial_filters // 1, 3, 1, activation='relu', bn=bn)
    #         x = self.conv2d(x, initial_filters // 1, 3, 1, activation='relu', bn=bn)

    #     if self.target_scale >= 4:
    #         x = self.upsampling(x)
    #         x = self.conv2d(x, initial_filters // 2, 3, 1, activation='relu', bn=bn)
    #         x = self.conv2d(x, initial_filters // 2, 3, 1, activation='relu', bn=bn)

    #     if self.target_scale >= 8:
    #         x = self.upsampling(x)
    #         x = self.conv2d(x, initial_filters // 4, 3, 1, activation='relu', bn=bn)
    #         x = self.conv2d(x, initial_filters // 4, 3, 1, activation='relu', bn=bn)

    #     if self.target_scale >= 16:
    #         x = self.upsampling(x)
    #         x = self.conv2d(x, initial_filters // 8, 3, 1, activation='relu', bn=bn)
    #         x = self.conv2d(x, initial_filters // 8, 3, 1, activation='relu', bn=bn)

    #     if self.target_scale >= 32:
    #         x = self.upsampling(x)
    #         x = self.conv2d(x, initial_filters // 16, 3, 1, activation='relu', bn=bn)
    #         x = self.conv2d(x, initial_filters // 16, 3, 1, activation='relu', bn=bn)

    #     g_output = self.conv2d(x, self.output_shape[-1], 1, 1, activation='sigmoid', bn=False)
    #     return g_input, g_output

    def build_g(self, bn):  # quicksr
        g_input = tf.keras.layers.Input(shape=self.input_shape)
        x = g_input
        x = self.conv2d(x, 32, 3, 1, activation='relu', bn=bn)
        x = self.conv2d(x, 32, 3, 1, activation='relu', bn=bn)
        x = self.conv2d(x, 32, 3, 1, activation='relu', bn=bn)
        x = self.conv2d(x, self.input_shape[-1] * self.target_scale * self.target_scale, 3, 1, activation='sigmoid', bn=bn)

        # use hard-coded constant for avoiding Not JSON Serializable error
        if self.target_scale == 2:
            g_output = tf.keras.layers.Lambda(lambda x: tf.nn.depth_to_space(x, 2), name='dn_output')(x)
        elif self.target_scale == 4:
            g_output = tf.keras.layers.Lambda(lambda x: tf.nn.depth_to_space(x, 4), name='dn_output')(x)
        return g_input, g_output

    def build_d(self, bn):
        d_input = tf.keras.layers.Input(shape=self.output_shape)
        x = d_input
        x = self.conv2d(x,  16, 3, 2, activation='leaky', bn=bn)
        x = self.conv2d(x,  32, 3, 2, activation='leaky', bn=bn)
        x = self.conv2d(x,  64, 3, 2, activation='leaky', bn=bn)
        x = self.conv2d(x, 128, 3, 2, activation='leaky', bn=bn)
        x = self.conv2d(x, 256, 3, 2, activation='leaky', bn=bn)
        x = self.flatten(x)
        d_output = self.dense(x, 1, activation='linear', bn=False)
        return d_input, d_output

    def conv2d(self, x, filters, kernel_size, strides, bn=False, activation='relu'):
        x = tf.keras.layers.Conv2D(
            strides=strides,
            filters=filters,
            padding='same',
            kernel_size=kernel_size,
            use_bias=not bn,
            kernel_initializer=self.kernel_initializer())(x)
        if bn:
            x = self.batch_normalization(x)
        return self.activation(x, activation)

    def dense(self, x, units, bn=False, activation='relu'):
        x = tf.keras.layers.Dense(
            units=units,
            use_bias=not bn,
            kernel_initializer=self.kernel_initializer())(x)
        if bn:
            x = self.batch_normalization(x)
        return self.activation(x, activation)

    def batch_normalization(self, x):
        return tf.keras.layers.BatchNormalization(momentum=0.8 if self.use_gan else 0.98)(x)

    def kernel_initializer(self):
        return tf.keras.initializers.RandomNormal(mean=0.0, stddev=0.02) if self.use_gan else 'glorot_normal'

    def activation(self, x, activation):
        if activation == 'leaky':
            x = tf.keras.layers.LeakyReLU(alpha=0.2)(x)
        else:
            x = tf.keras.layers.Activation(activation=activation)(x)
        return x

    def max_pool(self, x):
        return tf.keras.layers.MaxPool2D()(x)

    def upsampling(self, x):
        return tf.keras.layers.UpSampling2D()(x)

    def add(self, layers):
        return tf.keras.layers.Add()(layers)

    def flatten(self, x):
        return tf.keras.layers.Flatten()(x)

    def summary(self):
        self.g_model.summary()
        if self.use_gan:
            print()
            self.gan.summary()

