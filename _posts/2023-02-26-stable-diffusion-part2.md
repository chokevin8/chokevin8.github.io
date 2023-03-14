---
layout: post
title:  Latent/Stable Diffusion for Beginners! (Part 2)
date:   2023-02-25
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
- ### [Model Architecture](#model-architecture)
- ### [Experiments & Results](#experiment-results)
- ### [Variation of Stable Diffusion](#variation-stable-diffusion)

---

*Note: For Part 1, please click the link above in the table of contents.* 

<a id="stable-diffusion"></a>
##  **Stable Diffusion:**
Stable diffusion consists of three major components- an autoencoder, an U-Net, and a transformer. Each of the three components are critical and work together to work their magic. The autoencoder is responsible for two major tasks- 
the decoder allows the previously mentioned forward diffusion process to happen in the latent space. This means that the forward diffusion process, which is Markovian, would take less time since the image 
from our pixel space is essentially downsampled into the latent space. Without the autoencoder, the forward diffusion process would simply take too long. Likewise, the decoder is then responsible
for upsampling the generated latent space image back to the pixel space. With the help of the The U-Net, which is a popular model used in semantic segmentation, is a U-shaped convolutional neural network. It follows a U-shape because of 
its downsampling contracting path with max-pooling and its upsampling expansive path, with skip connections between each layers to preserve semantic details. The U-Net in stable diffusion, therefore, is responsible 
of...

insert image from: https://towardsdatascience.com/what-are-stable-diffusion-models-and-why-are-they-a-step-forward-for-image-generation-aa1182801d46



