# Super Resolution

Super Resolution refers to a task that uses deep learning to upscale the resolution of an image

SR is usually compared to bicubic interpolation, an image interpolation technique known for best performance in image upscaling

SR can learn high resolution textures that are not recognized by classic image interpolations such as bicubic

In some cases, using SR can help save processing time rather than performing bicubic interpolation

There are two ways to perform SR in this repository

The first is to give the model a loss directly so that the model can upscale the image and learn high resolution texture

This method is similar to training the AE and is simple, but works well with most data

If the results learned through the first method are not satisfactory, the SRGAN method allows the model to be trained in a different way with an additional adversarial attack loss term

Additionally learn the discriminator that classifies the model-generated and actual high resolution images, assume the SR model is a generator, and apply an adversarial attack

This way, unlike the first method, the model can create a more realistic image, but the disadvantage is that learning is unstable

To resolve this instability, you can specify the loss ignore threshold of the descriminator

Because most GAN problems arise because the discriminator is too powerful, ignore loss thresholds for discriminator can help train SRGAN

<img src="/md/srgan.jpg" width="435"><br>

## Result

We used the Korean license plate dataset and created an SR image of 192x96 size using 24x12 size images

Tested with a model consisting of 64 filters convolution layers that produces 8x higher resolution images

The order of the images is HR(High Resolution), LR(Low Resolution), Bicubic(Resize), SR(Super Resolution) from the left

<img src="/md/sample.jpg" width="800"><br>
