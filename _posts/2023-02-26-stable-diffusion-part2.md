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
### [Background](#background) ([Part 1- Previous Blog!](/blog/2023/stable-diffusion/))
- ###  [Introduction](#introduction)
- ### [Stable Diffusion vs GAN](#stable-diffusion-vs-gan)

### [Stable Diffusion](#stable-diffusion) (Part 2- This Blog!)
- ### [Model Architecture](#model-architecture)
- ### [Model Objective](#model-objective)
- ### [Experiments & Results](#experiment-results)
- ### [Applications of Stable Diffusion](#variation-stable-diffusion)

### [Math and Details Behind Stable Diffusion](#math-behind-stable-diffusion) ([Part 3- Next Blog!](/blog/2023/stable-diffusion-part3/))
- ### [Autoencoder](#autoencoder)
- ### [U-Net](#u-net)
- ### [Pretrained Encoder](#pretrained-encoder)

---

*Note: For Part 1 and 3 , please click the link above in the table of contents.* 

<a id="stable-diffusion"></a>
##  **Stable Diffusion:**
Stable diffusion consists of three major components- an autoencoder, a U-Net, and a pretrained encoder. Each of the three components are critical and work together to work their magic.
The entire model architecture and its three major components can be visualized in the below image:
<a id="model-architecture-objective"></a>
### **Model Architecture:**
<br>
<img src = "/assets/images/stable-diffusion.png" width = "523" height = "293" class = "center">
<figcaption>Diagram showing the general model architecture of the stable (latent) diffusion. Image credits to: [Link](https://towardsdatascience.com/what-are-stable-diffusion-models-and-why-are-they-a-step-forward-for-image-generation-aa1182801d46) 
</figcaption>
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
huge dataset of image-text pairs are used and thus allows text and image prompted stable diffusion as well (BERT could not handle images). 

Personally, I think that these huge pretrained encoders trained on massive datasets allowed the emergence of the diffusion model, as autoencoders and U-Nets were already in practice years ago.

<a id="model-objective"></a>
### **Model Objective:**

<a id="experiment-results"></a>
### **Experiments & Results:**


