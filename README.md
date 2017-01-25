# Colorizing-Images-Using-Adversarial-Networks
Given a grayscale photgraph as input, can we generate a *plausible* color version of the photograph?
Grayscale images could contain various degrees and intensities of colors, making this an ill-posed
problem. Therefore, we would not like to recover the ground truth colors, but rather generate plausible
colors.

#### Project Ideas/Goals
* Implement the [state of the art](http://richzhang.github.io/colorization/) for colorizing grayscale photos.
* [Use Generative Adversarial Networks](https://arxiv.org/pdf/1406.2661v1.pdf)
* Use [Energy-Based GANs](https://arxiv.org/pdf/1609.03126v3.pdf)
* Try also with normal GANs
* See if we can come up with varying colors by passing in a noise vector with the image.
* Can we pretrain the generator?
* Use the LAB colorspace as in the state of the art paper?
* Generate the channels (r,g,b) individually with multiple adversaries?
* Try and create a 'real time' variation that could colorize videos as they are playing.

**Important Note**: What we will essentially be learning is the transform from whatever function we use to
make the images gray to a colorized version, so we should also provide test results on "true" black and white
photos.

Need to think about how the generator will actually work. [This](http://richzhang.github.io/colorization/)
work explains why just taking the L2 norm between true and generated images produces
less saturated and more brown/green (averaged) images. Could use their approach for
the generator. I would also like to try generating color channels individually
and see what that yields.

#### Implementation Ideas
Use [Tensorflow](https://www.tensorflow.org/) with Python. They have pretty good [tutorials](https://www.tensorflow.org/tutorials/)

Data for us is essentially free because our "label" is the colorized image. I have 
[image-net](http://image-net.org/) downloaded so that is a great start. It is also the dataset used in the
baseline we have to compare to. It contains 1,281,167 training images, 100,000 test images, and 50,000
validation images.

I'm also currently downloading 

#### Useful links:

First paper on GANs

[Generative Adversarial Networks](https://arxiv.org/pdf/1406.2661v1.pdf)
___


If we go with Energy-Based GANs, this tutorial on energy based learning by Yann Lecunn is helpful.

[A Tutorial on Energy-Based Learning](http://yann.lecun.com/exdb/publis/pdf/lecun-06.pdf)
___

More from Lecunn on Energy-Based Models

[Loss Functions for Discriminative Training of Energy-Based Models](http://yann.lecun.com/exdb/publis/pdf/lecun-huang-05.pdf)
___

Tips and tricks for training GANs by Soumith Chintala (Facebook AI Research)

[Tips and Tricks for Training GANs](https://github.com/soumith/ganhacks)
___

All papers on GANs accepted for publication at NIPS 2016

[NIPS Accepted Papers for the Adversarial Training Workshop](https://sites.google.com/site/nips2016adversarial/home/accepted-papers)
___

Tutorial by Ian Goodfellow (wrote the first paper on GANs) given at NIPS 2016

[NIPS Tutorial on GANs](https://arxiv.org/pdf/1701.00160v3.pdf)
___

