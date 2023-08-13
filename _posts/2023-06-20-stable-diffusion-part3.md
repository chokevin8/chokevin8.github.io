---
layout: post
title:  Latent/Stable Diffusion for Beginners! (Part 3)
date:   2023-06-20
description: 
tags: deep-learning machine-learning latent-diffusion stable-diffusion generative-models
categories: posts
---
---

## **Table of Contents:**
### [Background](#background) ([Part 1](/blog/2023/stable-diffusion/))
- ###  [Introduction](#introduction)
- ### [Stable Diffusion vs GAN](#stable-diffusion-vs-gan)

### [Stable Diffusion](#stable-diffusion) ([Part 2](/blog/2023/stable-diffusion-part2/))
- ### [Motivation](#motivation)
- ### [Model Architecture](#model-architecture)
- ### [Experiments & Results](#experiment-results)

### [Math and Details Behind Stable Diffusion](#math-behind-stable-diffusion) (Part 3- This Blog!)
- ### [Model Objective](#model-objective)
- ### [Autoencoder](#autoencoder)
- ### [U-Net](#u-net)
- ### [Pretrained Encoder](#pretrained-encoder)

---

*Note: For Part 1 and 2, please click the link above in the table of contents.* 

In this last part of the blog, I will cover most of the important mathematical details behind latent diffusion that is necessary to fully understand
how it works. Let's first look at the model objective:

<a id="model-objective"></a>
##  **Model Objective:**
<p> 
In any model, there is a defined model loss function that is minimized- in the case of LDMs, since it has a tractable likelihood, it is 