# Colorizing-Images-Using-Adversarial-Networks
Given a grayscale photgraph as input, can we generate a *plausible* color version of the photograph?
Grayscale images could contain various degrees and intensities of colors, making this an ill-posed
problem. Therefore, we would not like to recover the ground truth colors, but rather generate plausible
colors.

#### Project Ideas/Goals
* Implement the [state of the art](http://richzhang.github.io/colorization/) for colorizing grayscale photos.
* [Use Generative Adversarial Networks](https://arxiv.org/pdf/1406.2661v1.pdf)
* Use [Energy-Based GANs](https://arxiv.org/pdf/1609.03126v3.pdf)
* Can we pretrain the generator?
* Use the LAB colorspace as in the state of the art paper?
* Generate the channels (r,g,b) individually with multiple adversaries?


Important Note: What we will essentially be learning is the transform from whatever function we use to make
the images gray to a colorized version, so we should also provide test results on "true" black and white
photos.

Can implement both energy based GANs and regular GANs, compare the two.

Need to think about how the generator will actually work. [This](http://richzhang.github.io/colorization/)
work explains why just taking the L2 norm between true and generated images produces
less saturated and more brown/green (averaged) images. Could use their approach for
the generator. I would also like to try generating color channels individually
and see what that yields.

Instead of using classification error, use the energy function as shown
[here](https://openreview.net/pdf?id=ryh9pmcee)


### Useful links:

[Generative Adversarial Networks](https://arxiv.org/pdf/1406.2661v1.pdf)

First paper on GANs
___

[A Tutorial on Energy-Based Learning](http://yann.lecun.com/exdb/publis/pdf/lecun-06.pdf)

Tutorial on energy based learning by Yann Lecunn
___

[Loss Functions for Discriminative Training of Energy-Based Models](http://yann.lecun.com/exdb/publis/pdf/lecun-huang-05.pdf)

More from Lecunn
___

[Tips and Tricks for Training GANs](https://github.com/soumith/ganhacks)

Tips and tricks for training GANs by Soumith Chintala (Facebook AI Research)
___

[NIPS Accepted Papers for the Adversarial Training Workshop](https://sites.google.com/site/nips2016adversarial/home/accepted-papers)

All papers on GANs accepted for publication at NIPS 2016
___

[NIPS Tutorial on GANs](https://arxiv.org/pdf/1701.00160v3.pdf)

Tutorial by Ian Goodfellow (wrote the first paper on GANs) given at NIPS 2016
___

