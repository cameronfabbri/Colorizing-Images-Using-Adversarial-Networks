# Colorizing-Images-Using-Adversarial-Networks
This is a project comparing multiple variations of Generative Adversarial Networks (GANs)
towards automatic image colorization.

### GAN Variations
* [Generative Adversarial Networks](https://arxiv.org/pdf/1406.2661v1.pdf)
* [Energy-Based GANs](https://arxiv.org/pdf/1609.03126v3.pdf)
* [Least Squares GANs](https://arxiv.org/pdf/1611.04076v2.pdf)
* [Wasserstein GAN](https://arxiv.org/pdf/1701.07875.pdf)

![jfkgray](http://i.imgur.com/0syARFb.png)
![jfkcol](http://i.imgur.com/LJ9Kkfk.png)

![aligray](http://i.imgur.com/B9S7FwL.png)
![alicol](http://i.imgur.com/7PaEtUd.png)

I used a combination of L1 and L2 loss along with the different GAN variations. I trained on the CelebA dataset, below shows some
results on the test set. From left to right: Input Image, GAN L1=100 L2=0, GAN L1=0 L2=1, LSGAN L1=100 L2=0, LSGAN L1=0 L2=1, EBGAN L1=100 L2=0,
EBGAN L1=0 L2=1, WGAN L1=100 L2=0, WGAN L1=0 L2=1, True Image.

![all](http://i.imgur.com/zXZr5iw.jpg)

Below show some results on photos outside the dataset with no true color version. From left to right: Input Image, GAN L1=100 L2=0, GAN L1=0 L2=1,
LSGAN L1=100 L2=0, LSGAN L1=0 L2=1, EBGAN L1=100 L2=0, EBGAN L1=0 L2=1, WGAN L1=100 L2=0, WGAN L1=0 L2=1

![allali](http://i.imgur.com/zhD5gsn.png)



#### Useful links:

[Colorful Image Colorization](https://arxiv.org/pdf/1603.08511.pdf)

[Let there be Color!](http://hi.cs.waseda.ac.jp/~/projects/colorization/data/colorization_sig2016.pdf)

[Learning Representations for Automatic Colorization](https://arxiv.org/pdf/1603.06668v1.pdf)

[Automatic Colorization of Grayscale Images](http://cs229.stanford.edu/proj2013/KabirzadehSousaBlaes-AutomaticColorizationOfGrayscaleImages.pdf)

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

