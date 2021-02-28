Link to the fully trained model: https://drive.google.com/drive/folders/12iBeCIruhnmoVyBdOXMlYrn7SMRVc7zN?usp=sharing

# Introduction

This tool was developed as a part of an [Outreachy][] (December
2020-March 2021) project proposal that aims to reduce vandalism attacks
on Wikimedia and its subsidiaries. The core of this tool is derived from
aspects of computer vision.
  
# The dataset

The dataset this deep learning tool was trained on was curated
personally to make sure there were no images that were wrongly
categorized or duplicates. The constituents of the dataset are:
follows Train Set (3262 questionable images and 3132 safe images),
Validation Set (1089 questionable images and 1048 safe images), and Test
Set (2127 images).

## Sources for the images

One of the sources for questionable media content was [this GitHub
repository][]. Other than that, popular websites such as [Imgur][] and
[Reddit][] was also referred to find different forms of
visual content that would adhere to either category.For safe images,
one of the primary sources was [Lorem Picsum][]. Apart from that,
websites such as [Shutterstock][] and [Flickr][] were also referred to.

## Image curation process

From the GitHub repository listed above, a list of URLs was retrieved
[this is the code][]. After this, duplicate images i.e. images with
different filenames but the same content were removed, which can also be
viewed on the same page. For safe images, one snippet of the scraping
process (which includes [Lorem Picsum][]), can be found [here][].
Duplicate images were removed in a fashion similar to that described
[here][this is the code].
  
# Design considerations and architecture details

The main objective that helped navigate through the development process
was to create a tool that is light-weight and robust but, at the same
time, does not compromise a lot on the observed accuracy. During the
research phase of this project, multiple recommended deep learning
models were tested on the curated dataset, such as Inception V3,
Inception V4, MobileNet V1[3], MobileNet V2, VGG16 and Resnet50. Upon
comparisons, MobileNet V1 was unanimously selected based on its
performance, size requirements, and processing time. Algorithms that
further reduce the size of a neural-network architecture were also
tested out (for example layer-based pruning and knowledge distillation
systems), but the performance of the resulting model was considerably
worse (although this subset model was a fraction of the size). Due to
time constraints, it was decided to proceed with the originally
developed neural-network model.

## Overview of the architecture

The core of this deep-learning-based tool is its pre-trained model -
MobileNet V1. The heart of the MobileNet V1 model is its usage of
depth-wise separable convolutions instead of the traditional
computationally expensive CNN-based operations. These split the scanning
kernel into two - one that assesses the spatial dimensions and another
that takes into account the depth of a specific layer. This makes a
considerable difference because the number of multiplication operations
is drastically reduced. This can be proven as follows:

<div class="proof">

*Proof.* Let *k*<sub>1</sub> × *k*<sub>1</sub> × *k*<sub>2</sub> be the
kernel size, and *n*<sub>*k*</sub> be the number of kernels. Let us say
that this kernel moves a total of *m* × *m* times. This leads to
*n*<sub>*k*</sub> × *k*<sub>1</sub> × *k*<sub>1</sub> × *k*<sub>2</sub> × *m* × *m*
multiplications, which can be written as
*k*<sub>1</sub><sup>2</sup>*m*<sup>2</sup>*n*<sub>*k*</sub>*k*<sub>2</sub>.
Alternatively, for the depth-based convolutional operation, the total
number of multiplications would be
*k*<sub>1</sub> × *k*<sub>1</sub> × *k*<sub>2</sub> × *m* × *m*, and for
the spatial dimensions based operation, the total number of
multiplications would be
*n*<sub>*k*</sub> × 1 × 1 × *k*<sub>2</sub> × *m* × *m*. Thus, the total
cost is
(*k*<sub>1</sub> × *k*<sub>1</sub> × *k*<sub>2</sub> × *m* × *m*) + (*n*<sub>*k*</sub> × 1 × 1 × *k*<sub>2</sub> × *m* × *m*).  
This can be written as
*k*<sub>1</sub><sup>2</sup>*m*<sup>2</sup>*k*<sub>2</sub> + *n*<sub>*k*</sub>*k*<sub>2</sub>*m*<sup>2</sup>
or
(*k*<sub>1</sub><sup>2</sup> + *n*<sub>*k*</sub>)*k*<sub>2</sub>*m*<sup>2</sup>.
As *k*<sub>1</sub> and *n*<sub>*k*</sub> are positive numbers with
*k*<sub>1</sub> ≤ *n*<sub>*k*</sub>,
*k*<sub>1</sub><sup>2</sup> + *n*<sub>*k*</sub> ≤ *k*<sub>1</sub><sup>2</sup>*n*<sub>*k*</sub>,
and hence, for all reasonable values of
*n*<sub>*k*</sub>, *k*<sub>1</sub>, *k*<sub>2</sub>, and *m*, depth-wise
separable convolutions are more efficient.

</div>

The MobileNet V1 has 28 layers (14 standard convolution layers, 13
depth-wise separable convolution layers, 1 average pool layer, 1 fully
connected layer, and a Softmax classifier in the end. ) At the end of
this architecture, a novel secondary architecture has been added that
allows the model to detect high-level features in our dataset for a more
personalized image classification operation.

## Time and Space Complexities

The MobileNet V1 architecture has around 4.2 million parameters, which
is considerably lesser than that of other models such as Resnet50 (∼ 23
a million parameters), Inception V3 (∼ 24 million parameters), and VGG16
(∼ 138 million parameters). Considering the number of parameters, the
space complexity for MobileNet V1 is one of the lowest among all
pre-trained deep learning models. Additionally, MobileNet V1 also uses
fewer multiplications and additions compared to other well-known
pre-trained deep learning models, which leads to a low time complexity
as well.

## Preliminary performance

Based on the curated dataset, after the training phase, the training
accuracy was 98.90%, the training error was 0.0346, the validation
accuracy was 96.43% and the validation loss was 0.1177.

## Modifying the architecture

As with most transfer-learning based approaches, the majority of the layers
in the architecture were frozen to facilitate faster training time and
to conserve the core weights that were developed in the original
MobileNet V1 model. However, it is fairly simple to "unfreeze" the
layers and make all of them trainable (or convert a greater portion of
the layers to trainable) if the core dataset can be improved on (or) if
more computational power is available. The secondary architecture (that
adds to the MobileNet V1 network) is fully malleable and, hence, new
variations can be tested out with ease.

# Installation and Implementation

## Requirements

## Acceptable media content extensions

This tool currently supports major image file extensions i.e. raster
image files (JPEG/JPG, PNG, and GIF). Support for PSD, SVG, and other
vector image formats might come shortly.

## Calling the API

## Integration with the abuse filter

# Future Work

## Image annotation tool

Currently, the [Labels][] tool developed by Wikimedia Labs helps users
participate in the task of text annotation that helps with the training
of intelligent wiki-tools based on Natural Language Processing. Future
work would involve creating a similar tool that allows users to assign
label(s) to images, which can then be used for bettering the
image-recognition based tools (for example this content filtration
tool) at Wikimedia.

## Video content filtering

Future versions of this tool could incorporate functionality that also
accepts videos and assesses the percentage of unsafe content in them.

## Categorization

This is subject to data availability. Deeper categories could be
introduced (for example: why is a particular content marked unsafe?) or
category-based tags could be assigned to each user input (for images
that might have been marked unsafe for multiple reasons).

  [Outreachy]: https://www.outreachy.org/
  [this GitHub repository]: https://github.com/EBazarov/nsfw_data_source_urls
  [Imgur]: https://imgur.com/
  [Reddit]: https://www.reddit.com/
  [Lorem Picsum]: https://picsum.photos/
  [Shutterstock]: https://www.shutterstock.com/
  [Flickr]: https://www.flickr.com/explore
  [this is the code]: https://github.com/HarshineeSriram/Outreachy_Wikimedia/blob/master/src/data-scrapers/scraper-unsafe-images.py
  [here]: https://github.com/HarshineeSriram/Outreachy_Wikimedia/blob/master/src/data-scrapers/scraper-safe-images.py
  [Labels]: https://labels.wmflabs.org/
