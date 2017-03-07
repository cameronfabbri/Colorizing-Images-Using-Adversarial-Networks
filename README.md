# Colorizing-Images-Using-Adversarial-Networks
Given a grayscale photgraph as input, can we generate a *plausible* color version of the photograph?
Grayscale images could contain various degrees and intensities of colors, making this an ill-posed
problem. Therefore, we would not like to recover the ground truth colors, but rather generate plausible
colors.

### Implementation
There will only be one train file, and will take in a configuration file defining what loss (type of GAN),
and what architecture to use.

#### Project Ideas/Goals
* Implement the [state of the art](http://richzhang.github.io/colorization/) for colorizing grayscale photos.
* [Use Generative Adversarial Networks](https://arxiv.org/pdf/1406.2661v1.pdf)
* Use [Energy-Based GANs](https://arxiv.org/pdf/1609.03126v3.pdf)
* Try also with normal GANs
* See if we can come up with varying colors by passing in a noise vector with the image.
* Can we pretrain the generator? Basically do the Colorization paper with a random seed
and the grayscale image, when it converges start training it on a GAN.
* Use the LAB colorspace as in the state of the art paper
* Could try pretraining every GAN before colorization, not even on colorization. Just train
it normally, then you will get all sorts of things for free when starting to train on color.
Not sure if you would have to alter z to be the same size as the input image, but we'll figure
that out.
can have some way of altering the color.

#### Things to compare
* State of the art colorization mentioned above.
* One or more state of the art general models (Alexnet, Inception, ResNet, etc)
* Energy-Based GANs
* Wasserstein GANs
* DCGANs
* Pretrained versions of the generator for the different GANs

Need to think about how the generator will actually work. [This](http://richzhang.github.io/colorization/)
work explains why just taking the L2 norm between true and generated images produces
less saturated and more brown/green (averaged) images. Could use their approach for
the generator. I would also like to try generating color channels individually
and see what that yields.


#### Useful links:

[Colorful Image Colorization](https://arxiv.org/pdf/1603.08511.pdf)

[Let there be Color!](http://hi.cs.waseda.ac.jp/~/projects/colorization/data/colorization_sig2016.pdf)

[Learning Representations for Automatic Colorization](https://arxiv.org/pdf/1603.06668v1.pdf)

[Automatic Colorization of Grayscale Images](http://cs229.stanford.edu/proj2013/KabirzadehSousaBlaes-AutomaticColorizationOfGrayscaleImages.pdf)



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

