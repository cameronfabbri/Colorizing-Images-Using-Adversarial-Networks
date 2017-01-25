# Colorizing-Images-Using-Adversarial-Networks
Colorizing images using an adversarial network approach.

Basically do what I did with the other colorization thing,
but after it's colorized, send it to the other network along
with the true color image, and have the adversary determine
which one is the true image and which one was colored by the
network.

Instead of using classification error, use the energy function as shown
[here](https://openreview.net/pdf?id=ryh9pmcee)


Useful links:

[Generative Adversarial Networks](https://arxiv.org/pdf/1406.2661v1.pdf)

First paper on GANs

[A Tutorial on Energy-Based Learning](http://yann.lecun.com/exdb/publis/pdf/lecun-06.pdf)

Tutorial on energy based learning by Yann Lecunn

[Loss Functions for Discriminative Training of Energy-Based Models](http://yann.lecun.com/exdb/publis/pdf/lecun-huang-05.pdf)

More from Lecunn

[Tips and Tricks for Training GANs](https://github.com/soumith/ganhacks)

Tips and tricks for training GANs by Soumith Chintala (Facebook AI Research)

[NIPS Accepted Papers for the Adversarial Training Workshop](https://sites.google.com/site/nips2016adversarial/home/accepted-papers)

All papers on GANs accepted for publication at NIPS 2016

[NIPS Tutorial on GANs](https://arxiv.org/pdf/1701.00160v3.pdf)

Tutorial by Ian Goodfellow (wrote the first paper on GANs) given at NIPS 2016

