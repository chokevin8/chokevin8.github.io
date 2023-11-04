---
layout: post
title:  Latent/Stable Diffusion Fully Explained! (Part 5)
date:   2023-11-10
description: Explanation of conditioning and classifier/classifier-free guidance.
tags: deep-learning machine-learning generative-models paper-review
categories: posts
---
---

## **Table of Contents:**
### [Latent/Stable Diffusion Fully Explained! (Part 1)](/blog/2023/stable-diffusion/)
- ### Introduction
- ### Stable Diffusion vs GAN

### [Latent/Stable Diffusion Fully Explained! (Part 2)](/blog/2023/stable-diffusion-part2/) 
- ### Motivation
- ### Model Architecture
- ### Experiments & Results

### [Latent/Stable Diffusion Fully Explained! (Part 3)](/blog/2023/stable-diffusion-part3/) 
- ### VAEs and ELBO
- ### Model Objective

### [Latent/Stable Diffusion Fully Explained! (Part 4)](#stable-diffusion-in-numbers-2) (This Blog!)
- ### [Different View on Model Objective](#model-objective2)
- ### [Training and Inference](#training-inference)

### [Latent/Stable Diffusion Fully Explained! (Part 5)](/blog/2023/stable-diffusion-part5/)
- ### Conditioning 
- ### Classifier-Free Guidance
- ### Summary

---

*Note: For other parts, please click the link above in the table of contents.* 

<a id="stable-diffusion-in-numbers-2"></a>
## **Stable Diffusion In Numbers Continued**



<a id="conditioning"></a>
###  ***Conditioning:***
For conditioning, look at table 15 of LDM paper
maybe include autoencoder training as well since conditioning is basically UNet training details + pretrained encoder.
Autoencoder training is in appendix G: Details on Autoencoder Models

<a id="classifier-free-guidance"></a>
###  ***Classifier-Free Guidance:***

<a id="summary"></a>
###  ***Summary:***