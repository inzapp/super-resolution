"""
Authors : inzapp

Github url : https://github.com/inzapp/super_resolution

Copyright 2023 inzapp Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License"),
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

     http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""
import cv2
import numpy as np
from concurrent.futures.thread import ThreadPoolExecutor


class DataGenerator:
    def __init__(self,
                 generator,
                 image_paths,
                 input_shape,
                 output_shape,
                 batch_size,
                 dtype='float32'):
        self.generator = generator
        self.image_paths = image_paths
        self.input_shape = input_shape
        self.output_shape = output_shape
        self.batch_size = batch_size
        self.half_batch_size = batch_size // 2
        self.dtype = dtype
        self.pool = ThreadPoolExecutor(8)
        self.img_index = 0
        np.random.shuffle(self.image_paths)

    def __len__(self):
        return int(np.floor(len(self.image_paths) / self.batch_size))

    def load(self, use_gan):
        if use_gan:
            from super_resolution import SuperResolution
            z = self.preprocess_images(self.load_images(count=self.batch_size, shape=self.input_shape, interpolation='area'))
            real_dx = self.preprocess_images(self.load_images(count=self.half_batch_size, shape=self.output_shape, interpolation='auto'))
            real_dy = np.ones((self.half_batch_size, 1), dtype=self.dtype)
            fake_dx = np.asarray(SuperResolution.graph_forward(model=self.generator, x=z[:self.half_batch_size])).astype(self.dtype)
            fake_dy = np.zeros((self.half_batch_size, 1), dtype=self.dtype)
            dx = np.append(real_dx, fake_dx, axis=0)
            dy = np.append(real_dy, fake_dy, axis=0)
            gx = z
            gy = np.append(real_dy, real_dy, axis=0)
            return dx, dy, gx, gy
        else:
            gy = self.preprocess_images(self.load_images(count=self.batch_size, shape=self.output_shape, interpolation='auto'))
            gx = self.resize_images(images=gy, size=(self.input_shape[1], self.input_shape[0]), interpolation='area')
            return None, None, gx, gy

    def next_image_path(self):
        path = self.image_paths[self.img_index]
        self.img_index += 1
        if self.img_index == len(self.image_paths):
            self.img_index = 0
            np.random.shuffle(self.image_paths)
        return path

    def load_image(self, image_path, channels):
        img = cv2.imdecode(np.fromfile(image_path, dtype=np.uint8), cv2.IMREAD_GRAYSCALE if channels == 1 else cv2.IMREAD_COLOR)
        if img is None:
            print(f'img is None, path : {image_path}')
            return img
        if channels == 1 and len(img.shape) == 2:
            img = img.reshape(img.shape + (1,))
        img = np.asarray(img).astype('uint8')
        return img

    def load_images(self, count, shape, interpolation='auto'):
        fs = []
        for _ in range(count):
            fs.append(self.pool.submit(self.load_image, self.next_image_path(), shape[-1]))
        images = []
        for f in fs:
            img = f.result()
            img = self.resize(img, (shape[1], shape[0]), interpolation)
            if shape[-1] == 1 and len(img.shape) == 2:
                img = img.reshape(img.shape + (1,))
            images.append(img)
        return images

    def resize(self, img, size, interpolation):
        assert interpolation in ['nearest', 'area', 'bicubic', 'auto', 'random']
        interpolation_cv = None
        img_h, img_w = img.shape[:2]
        if interpolation == 'nearest':
            interpolation_cv = cv2.INTER_NEAREST
        elif interpolation == 'area':
            interpolation_cv = cv2.INTER_AREA
        elif interpolation == 'bicubic':
            interpolation_cv = cv2.INTER_CUBIC
        elif interpolation == 'auto':
            if size[0] == img_w and size[1] == img_h:
                interpolation_cv = cv2.INTER_LINEAR
            elif size[0] > img_w or size[1] > img_h:
                interpolation_cv = cv2.INTER_CUBIC
            else:
                interpolation_cv = cv2.INTER_AREA
        else:
            interpolation = np.random.choice([cv2.INTER_LINEAR, cv2.INTER_AREA, cv2.INTER_NEAREST, cv2.INTER_CUBIC])
        return cv2.resize(img, size, interpolation=interpolation_cv)

    def resize_images(self, images, size, interpolation='auto'):
        ret = []
        for image in images:
            ret.append(self.resize(image, size, interpolation))
        return np.asarray(ret)

    def preprocess(self, img):
        if img.shape[-1] == 3:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # rb swap
        x = np.asarray(img).astype(self.dtype) / 255.0
        return x

    def preprocess_images(self, images):
        ret = []
        for image in images:
            ret.append(self.preprocess(image))
        return np.asarray(ret)

    def postprocess(self, output):
        img = np.clip(np.asarray(output) * 255.0, 0.0, 255.0).astype('uint8')
        if img.shape[-1] == 3:
            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)  # rb swap
        return img

    def postprocess_images(self, outputs):
        ret = []
        for output in outputs:
            ret.append(self.postprocess(output))
        return np.asarray(ret)

