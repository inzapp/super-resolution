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
import argparse

from super_resolution import TrainingConfig, SuperResolution


if __name__ == '__main__':
    config = TrainingConfig(
        train_image_path=r'/train_data/imagenet/train',
        validation_image_path=r'/train_data/imagenet/validation',
        model_name='model',
        input_shape=(32, 32, 1),
        target_scale=2,
        lr=0.001,
        warm_up=0.5,
        batch_size=4,
        view_grid_size=4,
        save_interval=10000,
        iterations=1000000,
        use_gan=False,
        training_view=False)

    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='', help='pretrained model path')
    parser.add_argument('--predict', action='store_true', help='predict images using given path dataset')
    parser.add_argument('--evaluate', action='store_true', help='calculate psnr, ssim score with given dataset')
    parser.add_argument('--path', type=str, default='', help='image or video path for prediction or evaluation')
    parser.add_argument('--dataset', type=str, default='validation', help='dataset for evaluate, train or validation available')
    parser.add_argument('--save-count', type=int, default=0, help='count for save images')
    args = parser.parse_args()
    if args.model != '':
        config.pretrained_model_path = args.model
    if args.evaluate or args.predict:
        config.use_fixed_seed = True
    super_resolution = SuperResolution(config=config)
    if args.evaluate:
        super_resolution.evaluate(image_path=args.path, dataset=args.dataset)
    elif args.predict:
        super_resolution.evaluate(image_path=args.path, dataset=args.dataset, show_image=args.save_count==0, save_count=args.save_count)
    else:
        SuperResolution(config=config).train()

