---
layout: post
title:  Latent/Stable Diffusion for Beginners! (Part 4)
date:   2023-08-25
description: 
tags: deep-learning machine-learning generative-models paper-review
categories: posts
---
---

## **Table of Contents:**
### [Background (Part 1)](/blog/2023/stable-diffusion/))
- ### Introduction
- ### Stable Diffusion vs GAN

### [Stable Diffusion In Words (Part 2)](/blog/2023/stable-diffusion-part2/) 
- ### Motivation
- ### Model Architecture
- ### Experiments & Results

### [Stable Diffusion In Numbers (Part 3)](/blog/2023/stable-diffusion-part3/) 
- ### VAEs and ELBO
- ### Model Objective

### [Stable Diffusion In Numbers Continued (Part 4)](#stable-diffusion-in-numbers-2) (This Blog!)
- ### [Training and Inference](#training-inference)
- ### [Conditioning](#conditioning)
- ### [Classifier-Free Guidance](#classifier-free-guidance)

---

<a id="stable-diffusion-in-numbers-2"></a>
## **Stable Diffusion In Numbers Continued**

In this last part of the blog, I want to cover the mathematical details of conditioning and also classifier-free guidance. Before, that let's briefly
look at the algorithms for training and inference.

<a id="training-inference"></a>
###  ***Training and Inference:***

Now that we've derived the training (loss) objective, let's briefly go over the entire training and the inference algorithm, look below:

<img src = "/assets/images/train_inference_algorithm.png" width = "985" height = "250" class = "center">
<figcaption>The training and inference algorithm, summarized.</figcaption>
<br>

Let's first look at the training algorithm:
1. We repeat the below process (steps 2~5) until convergence or a preset number of epochs. 
2. Sample an image $$x_0$$ from our dataset/data distribution, $$q(x_0)$$.
3. Sample t, or timestep.
4. Sample noise from a normal distribution $$\epsilon \sim \mathcal{N}(0, I)$$
5. Take gradient descent step on the previous training objective $$L_{LDM} = ||\epsilon - \epsilon_{\theta}(x_t,t)||^2 $$ with respect to $$\theta$$, which is the 
parameters of the weights and biases of the decoder.
 
We are now well familiar with the training process since the objective was already explained in the previous part of the blog post. 

The above sampling algorithm is the DDPM sampling process, which is just the reverse diffusion process explained in the previous part. 

Now, note that for sampling, we only need the trained decoder from above (no encoder). Therefore, we sample latent noise $$x_T$$ from prior $$p(x_T)$$, which is $$\epsilon \sim \mathcal{N}(0, I)$$
and then run the series of $$T$$ equally weighted autoencoders as mentioned before in a Markovian style (sample from $$x_{t-1}$$). However, the sampling process using
Denoising Diffusion Probabilistic Model (DDPM) uses a Markovian sampling process while an improved method called Denoising Diffusion Implicit Model (DDIM) uses a non-Markovian
sampling process that makes the process much quicker. Therefore, DDIM uses $$S$$ steps instead of $$T$$ where $$ S<T $$, and the authors of LDM therefore use DDIM over DDPM.




To derive the DDIM sampling process, we utilize the *reparametrization trick*, which we applied in equation #5 above. The reparametrization trick is used whenever we sample from
a distribution (Gaussian in our case) that is not directly differentiable. For our case, the mean and the variance of the distribution are both dependent on the model
parameters, which is learned through SGD (as shown above). The issue is that because sampling from the Gaussian distribution is stochastic, we cannot compute the gradient anymore to update
the mean and variance parameters. So, we introduce the auxiliary random variable $$\epsilon$$ that is deterministic since it is sampled from a fixed standard Gaussian distribution ($$\epsilon \sim \mathcal{N}(0, 1) $$),
which allows SGD to be possible since $$\epsilon$$ is not dependent on the model parameters. Therefore, the reparametrization trick $$ x = \mu + \sigma * \epsilon$$ works by initially computing the mean and standard deviation using current weights given input data,
then drawing deterministic random variable $$\epsilon$$ to obtain the desired sample $$x$$. Then, loss can be computed with respect to mean and variance, and they can be backpropagated via SGD.

Now, the previous reparametrization trick was used to allow SGD, but this time we can also use the reparametrization trick to essentially alter our sampling process $$q(x_{t-1}|x_t,x_0)$$ to be parametrized by another random variable,
a desired standard deviation $$\epsilon_t$$. The reparametrization is shown below:

The main advantages of DDIM over DDPM are:

1. Consistency: DDIMs are consistent, meaning that if we initialize the same latent variable $$x_T$$ via same random seed during sampling, the samples 
2. 




For conditioning, look at table 15 of LDM paper
*Note: For other parts, please click the link above in the table of contents.* 
maybe include autoencoder training as well since conditioning is basically UNet training details + pretrained encoder.
Autoencoder training is in appendix G: Details on Autoencoder Models