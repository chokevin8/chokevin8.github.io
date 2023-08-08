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
In part 1, I've introduced the concept of generative models and the disadvantages of GANs, but that's not the true motivation of the authors of the paper.
Diffusion models are a (explicit) likelihood-based model, which can be classified as similar types to an autoregressive or a normalizing flow model. 
<br>
Therefore, diffusion models tend to share similar disadvantages as autoregressive or normalizing-flow models, as it suffers from high computational cost due to the model spending
a lot of its resources and times preserving the details of the data entirely in the pixel space, which are often times unnecessary. Therefore, the authors aim to combat 
this issue by enabling diffusion models to train in the latent space without loss of performance, which enables training on limited computational resources.

Now, converting the input image to the latent space isn't an easy task, as it requires compressing the images without losing its perceptual and semantic details.
*Perceptual image compression* is just like what it sounds- it aims to preserve the visual quality of an image by prioritizing the information that is most noticeable to human 
perception while compressing and removing parts of the image that is less sensitive to human perception. In most cases, high frequency components of the image, which tend to 
be rapid changes in the images like edges and fine patterns can be removed as human perception is more sensitive to changes in low frequency components of the image
such as major shapes and structures.
<br>
*Semantic image compression* is a bit different- it aims to preserve the high-level semantic information that is important for understanding
the overall image content, such as the outlines and borders of key objects of the image. 

In stable diffusion, the authors perform perceptual and semantic compression in two distinct steps:
1. The autoencoder (model architecture is explained in the next section) converts and compresses the input image from the pixel space to the latent space without losing perceptual details. Essentially, the autoencoder performs
the perceptual compression by removing high-frequency details. 
2. The pretrained encoder and the U-Net, which as a combination is responsible for the actual generation of images (reverse diffusion), learns the semantic composition
of the data. Essentially, the pretrained encoder and the U-Net performs semantic compression by learning how to generate the high-level semantic information of the input image.

Now, one of the biggest reasons why previous diffusion or likelihood-based models required such high computational cost was because the models spent so much time processing
losses, gradients, 

Therefore, the authors hypothesize that:
<ul>
    <li> 1. asfdasfasf </li>
</ul>

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
to happen in the latent space.

2. **U-Net:** The U-Net, which is a popular model used in semantic segmentation, is a U-shaped convolutional neural network. It follows a U-shape because of its downsampling contracting path with max-pooling and its upsampling expansive path, 
with skip connections between each layers to preserve semantic details. The U-Net in stable diffusion, is responsible for denoising the noisy latent vector that was produced by 
the encoder, meaning that it is responsible for the reverse diffusion process. Therefore, when we train a diffusion model, we're essentially training the denoising U-Net. 
However, this denoising U-Net isn't without additional help, as it is conditioned by not only the noisy latent vector, but also by the latent embeddings generated
by the encoder via a cross-attention mechanism built in to the U-Net architecture. Since it is a cross-attention mechanism, these embeddings can be from different modalities such as text, image, semantic map, and more. 

3. **Pretrained Encoder:** 
<br>
Therefore, the pretrained text/image (mostly texts and images are used) encoder is responsible for projecting the conditioning text/image prompts to an intermediate representation that can be
mapped to the cross-attention components, which is the usual query, key, and value matrices. For example, BERT or BERT-like pretrained text encoders have been pretrained on huge corpus datasets and
are suited to take text prompts and generate token embeddings which are the intermediate representations that gets mapped to the cross-attention components. Nowadays, CLIP or CLIP-like pretrained text/image encoders pretrained on
huge dataset of image-text pairs are used and thus allows text and image prompted stable diffusion as well (BERT can not handle images). 

Now, with the above motivation resulting in the authors designing this unique model architecture, the authors performed several experiments to verify their claims.

<a id="experiment-results"></a>
### **Experiments & Results:**

<a id="applications-stable-diffusion"></a>
### **Applications of Stable Diffusion:**

*Image credits to: [Stable Diffusion Architecture](https://towardsdatascience.com/what-are-stable-diffusion-models-and-why-are-they-a-step-forward-for-image-generation-aa1182801d46) 