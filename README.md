# ERRNet

The implementation of CVPR 2019 paper "Single Image Reflection Removal Exploiting Misaligned Training Data and Network Enhancements" [arxiv](https://arxiv.org/abs/1904.00637)


## Highlights

* Our network can extract the background image layer devoid of reflection artifacts, as in the example:

<img src="imgs/animation2.gif" height="140px"/> <img src="imgs/animation1.gif" height="140px"/> 

* We captured a new dataset containing 450 unaligned image pairs that are considerably easier to collect.
Image samples from our unaligned dataset are shown below:

<img src="imgs/unaligned1.gif" height="140px"/> <img src="imgs/datacollection_ours.jpg" height="140px"/>  <img src="imgs/unaligned2.gif" height="140px"/> 

* We introduce a simple but powerful alignment-invariant loss function to facilitate exploiting misaligned real-world training data. Finetuning on unaligned image pairs with our loss leads to sharp and reflection-free results, in contrast to the blurry ones when using a conventional pixel-wise loss (L1, L2, e.t.c.). The resulting images finetuned by different losses are shown below: (Left: Pixel-wise loss; Right: Ours)

<img src="imgs/unaligned_pixel.gif" height="140px"/> <img src="imgs/unaligned_ours.gif" height="140px"/>   


## Prerequisites
* Python >=3.5, PyTorch >= 0.4.1
* Requirements: opencv-python, tensorboardX, visdom
* Platforms: Ubuntu 16.04, cuda-8.0

## Citation

If you find our code helpful in your research or work please cite our paper.

```
 @inproceedings{wei2019single,
   title={Single Image Reflection Removal Exploiting Misaligned Training Data and Network Enhancements},
   author={Wei, Kaixuan and Yang, Jiaolong and Fu, Ying and David, Wipf and Huang, Hua},
   booktitle={IEEE Conference on Computer Vision and Pattern Recognition},
   year={2019},
 }
```

## Acknowledgments
Code architecture is inspired by [CycleGAN](https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix) and [EDSR](https://github.com/thstkdgus35/EDSR-PyTorch). 
