# Colorizing-Images-Using-Adversarial-Networks
This is a project comparing multiple variations of Generative Adversarial Networks (GANs)
towards automatic image colorization, currently [in progress](http://irvlab.dl.umn.edu/projects/adversarial-image-colorization)

### GAN Variations
* [Generative Adversarial Networks](https://arxiv.org/pdf/1406.2661v1.pdf)
* [Energy-Based GANs](https://arxiv.org/pdf/1609.03126v3.pdf)
* [Least Squares GANs](https://arxiv.org/pdf/1611.04076v2.pdf)
* [Wasserstein GAN](https://arxiv.org/pdf/1701.07875.pdf)

![jfkgray](http://i.imgur.com/0syARFb.png)
![jfkcol](http://i.imgur.com/LJ9Kkfk.png)

![aligray](http://i.imgur.com/7bjJt0n.png)
![alicol](http://i.imgur.com/7PaEtUd.png)

I used a combination of L1 and L2 loss along with the different GAN losses. Full details can be found in the [write up](https://github.com/cameronfabbri/Colorizing-Images-Using-Adversarial-Networks/blob/master/files/report.pdf).
I trained on the CelebA dataset because it's only a single class, and I was really looking if this would even work at all.
Below shows some results on the test set. From left to right: Input Image, GAN L1=100 L2=0,
GAN L1=0 L2=1, LSGAN L1=100 L2=0, LSGAN L1=0 L2=1, EBGAN L1=100 L2=0, EBGAN L1=0 L2=1, WGAN L1=100 L2=0, WGAN L1=0 L2=1, True Image.

![all](http://i.imgur.com/zXZr5iw.jpg)

Below show some results on photos outside the dataset with no true color version. From left to right: Input Image, GAN L1=100 L2=0, GAN L1=0 L2=1,
LSGAN L1=100 L2=0, LSGAN L1=0 L2=1, EBGAN L1=100 L2=0, EBGAN L1=0 L2=1, WGAN L1=100 L2=0, WGAN L1=0 L2=1

![allali](http://i.imgur.com/zhD5gsn.png)

Currently training on Imagenet to see how it works on the multiclass scenario.

#### Useful links:

[Colorful Image Colorization](https://arxiv.org/pdf/1603.08511.pdf)

[Let there be Color!](http://hi.cs.waseda.ac.jp/~/projects/colorization/data/colorization_sig2016.pdf)

[Learning Representations for Automatic Colorization](https://arxiv.org/pdf/1603.06668v1.pdf)

[Automatic Colorization of Grayscale Images](http://cs229.stanford.edu/proj2013/KabirzadehSousaBlaes-AutomaticColorizationOfGrayscaleImages.pdf)

