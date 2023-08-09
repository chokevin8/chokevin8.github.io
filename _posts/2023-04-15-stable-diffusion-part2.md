---
layout: post
title:  Latent/Stable Diffusion for Beginners! (Part 2)
date:   2023-04-15
description: 
tags: deep-learning machine-learning latent-diffusion stable-diffusion generative-models
categories: posts
---
---

## **Table of Contents:**
### [Background](#background) ([Part 1](/blog/2023/stable-diffusion/))
- ###  [Introduction](#introduction)
- ### [Stable Diffusion vs GAN](#stable-diffusion-vs-gan)

### [Stable Diffusion](#stable-diffusion) (Part 2- This Blog!)

- ### [Motivation](#motivation)
- ### [Model Architecture](#model-architecture)
- ### [Experiments & Results](#experiment-results)
- ### [Applications of Stable Diffusion](#applications-stable-diffusion)

### [Math and Details Behind Stable Diffusion](#math-behind-stable-diffusion) ([Part 3](/blog/2023/stable-diffusion-part3/))
- ### [Model Objective](#model-objective)
- ### [Autoencoder](#autoencoder)
- ### [U-Net](#u-net)
- ### [Pretrained Encoder](#pretrained-encoder)

---

*Note: For Part 1 and 3 , please click the link above in the table of contents.* 

<a id="motivation"></a>
##  **Motivation:**
<p> 
In part 1, I've introduced the concept of generative models and the disadvantages of GANs, but that's not the true motivation of the authors of the paper.
Diffusion models are a (explicit) likelihood-based model, which can be classified as similar types to an autoregressive or a normalizing flow model. 
</p>
<p>
Therefore, diffusion models tend to share similar disadvantages as autoregressive or normalizing-flow models, as it suffers from high computational cost due to the model spending
a lot of its resources and times preserving the details of the data entirely in the pixel space, which are often times unnecessary. Therefore, the authors aim to combat 
this issue by enabling diffusion models to train in the latent space without loss of performance, which enables training on limited computational resources.
</p>
<p>

Now, converting the input image to the latent space isn't an easy task, as it requires compressing the images without losing its perceptual and semantic details.
*Perceptual image compression* is just like what it sounds- it aims to preserve the visual quality of an image by prioritizing the information that is most noticeable to human 
perception while compressing and removing parts of the image that is less sensitive to human perception. In most cases, high frequency components of the image, which tend to 
be rapid changes in the images like edges and fine patterns can be removed as human perception is more sensitive to changes in low frequency components of the image
such as major shapes and structures.
</p>
<p>
*Semantic image compression* is a bit different- it aims to preserve the high-level semantic information that is important for understanding
the overall image content, such as the outlines and borders of key objects of the image. 
</p>

In stable diffusion, the authors perform perceptual and semantic compression in two distinct steps:
1. The autoencoder (model architecture will be explained in the next section) converts and compresses the input image from the pixel space to the latent space without losing perceptual details. Essentially, the autoencoder performs
the perceptual compression by removing high-frequency details. 
2. The pretrained encoder and the U-Net, which as a combination is responsible for the actual generation of images (reverse diffusion), learns the semantic composition
of the data. Essentially, the pretrained encoder and the U-Net performs semantic compression by learning how to generate the high-level semantic information of the input image.

<p>
The autoencoder that maps between the pixel and the latent space and performs perceptual compression is called the "latent diffusion model". However, it seems like the authors meant to say that
if the entire model architecture contains this mapping autoencoder, it can be under the same class of latent diffusion models. The authors mention that this universal autoencoding stage is needed
to be trained only once, and this autoencoder can be utilized in various different multi-modal tasks.
</p>
Now, one of the biggest reasons why previous diffusion or likelihood-based models required such high computational cost was because the models spent so much time updating and calculating
losses, gradients, and different weights of the backbone during training and inference on parts of the images that are not important perceptual details.
<br>
<img src = "/assets/images/distortion_vs_rate.png" width = "800" height = "500" class = "center">
<figcaption>Diagram showing the relationship between rate and distortion and its tradeoff.</figcaption>
<br>
<p>

As seen above in the graph from the paper, we see the rate-distortion tradeoff. *Distortion* can be thought of as the root-mean-squared error (RMSE) between
the original input image and the final generated image from the decoder. The lower the distortion, the lower the root-mean-squared error between the 
original image and the generated image. One may erroneously assume that a low distortion would always mean good perceptual quality of the image, but this is
actually the complete opposite- optimizing one will alway come at the expense of another.
</p>
<p>

<i><b>Rate</b></i>, or bits per dimension or pixel, can be thought of as the amount of information. Therefore, higher the rate, the more "information" there is in the image. I believe
that the diagram shows the progression of the reverse diffusion process, where it starts at high distortion and zero rate (completely noised image) at time T, and where it ends
at low distortion and high rate (completely denoised image) at time 0. Thus, it makes sense why distortion would decrease when rate is increased, as shown in the graph. 
The graph above first shows the separately trained universal autoencoder for an effective mapping of the input image in the pixel space to the latent space. 
Therefore, this autoencoder allows the reverse diffusion process of the conditioned U-Net to focus on generating the semantic details for semantic compression.  
</p>

<p>

But one may ask, *why specifically use U-Net?* The authors believe that by using U-Net, they can utilize the inductive bias of U-Net to generate high quality
images. This is indeed true, as U-Nets, like other convolution-based networks, naturally excel at capturing the spatial relationship of images and have unique
advantages like translational invariance due to convolutions. 
</p>

Therefore, the authors state that their work offers three main advantages over other previous diffusion models or any general generative models:
<ul>
    <li> Training an autoencoder that maps pixel to latent space makes the forward and reverse diffusion process computationally efficient with minimal losses in perceptual quality. </li>
    <li> The autoencoder only needs to be trained once and then can be used for various downstream training of multiple, multi-modal generative
models.</li>
    <li> By utilizing the inductive bias of U-Net, the model does not need aggressive compression of images which could deteriorate perceptual quality of generated images. </li>
</ul>

Before looking at experiments and results the authors conclueded to verify those claims above, let's look at the overall model architecture first.
---

<a id="model-architecture-objective"></a>
### **Model Architecture:**
Stable diffusion consists of three major components- an autoencoder, a U-Net, and a pretrained encoder. Each of the three components are critical and work together to work their magic.
The entire model architecture and its three major components can be visualized in the below image:

<br>
<img src = "/assets/images/stable-diffusion.png" width = "800" height = "500" class = "center">
<figcaption>Diagram showing the general model architecture of the stable (latent) diffusion.</figcaption>
<br>

1. **Autoencoder:** The autoencoder is responsible for two major tasks, with the encoder and the decoder being responsible for each task. First, the encoder allows the previously mentioned forward
diffusion process to happen in the latent space. This means that the forward diffusion process, which is Markovian, would take less time since the image from our pixel space is 
essentially downsampled into the latent space. Without the encoder, the forward diffusion process in the pixel space would simply take too long. Likewise, the decoder is then responsible
for upsampling the generated latent space image back to the pixel space. The generated latent space image is obtained from the output of the U-Net, which will be mentioned next. 
The decoder is needed because the latent space image needs to be converted back to the pixel space to obtain our desired image. Basically, the autoencoder allows the forward and the backward diffusion process
to happen in the latent space, and also performs perceptual compression by removing high-frequency details as explained in the previous section. Note that this autoencoder can be separately trained only once and
be applied in various different tasks since the generative power of the model resides in the U-Net + pretrained encoder.

2. **U-Net:** The U-Net, which is a popular model used in semantic segmentation, is a U-shaped fully convolutional neural network. It follows a U-shape because of its downsampling contracting path with max-pooling and its upsampling expansive path, 
with skip connections between each layers to preserve semantic details. The U-Net in stable diffusion, is responsible for denoising the noisy latent vector that was produced by 
the encoder, meaning that it is responsible for the reverse diffusion (generation) process. Therefore, when we train a diffusion model, we're essentially training the denoising U-Net. 
However, this denoising U-Net isn't without additional help, as it is conditioned by not only the noisy latent vector, but also by the latent embeddings generated
by the encoder via a cross-attention mechanism built in to the U-Net architecture. Since it is a cross-attention mechanism, these embeddings can be from different modalities such as text, image, semantic map, and more. 
The inductive bias of the U-Net giving it a natural advantage in handling image data with cross-attention in the backbone makes the generative process stable (as aggressive compression is not needed) and 
flexible (cross attention allows multi-modality).

3. **Pretrained Encoder:** 
<br>
Therefore, the pretrained text/image (mostly text and image modalities are used as prompts) encoder is responsible for projecting the conditioning text/image prompts to an intermediate representation that can be
mapped to the cross-attention components, which is the usual query, key, and value matrices. For example, BERT or BERT-like pretrained text encoders have been pretrained on huge corpus datasets and
are suited to take text prompts and generate token embeddings which are the intermediate representations that gets mapped to the cross-attention components. Nowadays, CLIP or CLIP-like pretrained text/image encoders pretrained on
huge dataset of image-text pairs are used and thus allows text and image prompted stable diffusion as well (BERT can not handle images). 

Now, with the above motivation resulting in the authors designing this unique model architecture, the authors performed several experiments to verify their claims.

<a id="experiment-results"></a>
### **Experiments & Results:**


<a id="applications-stable-diffusion"></a>
### **Applications of Stable Diffusion:**

*Image credits to: [Stable Diffusion Architecture](https://towardsdatascience.com/what-are-stable-diffusion-models-and-why-are-they-a-step-forward-for-image-generation-aa1182801d46) 