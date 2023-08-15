---
layout: post
title:  Latent/Stable Diffusion for Beginners! (Part 3)
date:   2023-06-20
description: 
tags: deep-learning machine-learning latent-diffusion stable-diffusion generative-models autoencoders u-net pretrained-encoders variational-autoencoders 
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

### [Stable Diffusion In Numbers (Part 3)](#stable-diffusion-in-numbers-1) (This Blog!)
- ### [VAEs and ELBO](#vaes-elbo)
- ### [Model Objective](#model-objective)

### [Stable Diffusion In Numbers Continued (Part 4)](/blog/2023/stable-diffusion-part4/)
- ### Autoencoder
- ### U-Net
- ### Pretrained Encoder

---

*Note: For other parts, please click the link above in the table of contents.* 

<a id="stable-diffusion-in-numbers-1"></a>
## **Stable Diffusion In Numbers**
In this part of the blog, I will cover the mathematical details behind latent diffusion that is necessary to fully understand
how latent diffusion works. Before looking at the model objective of LDMs, I think it's important to do an in-depth review on VAEs and how the Evidence Lower Bound
(ELBO) is utilized:

<a id="vaes-elbo"></a>
###  ***VAEs and ELBO:***

Let's look at variational autoencoders (VAEs) in a probabilistic way. The variational autoencoder holds a probability model with the $x$ representing
the data, and the $z$ representing the latent variables of the autoencoder. Remember that we want our latent variable $$z$$ to model the data $$x$$ as 
accurately as possible. Note that $$x$$ can be seen, but $$z$$ cannot since it is in the latent space. To perform the generative process, or run inference, 
for each individual data $$j$$, we first sample latent variable $$z_i$$ from the prior $$P(z)$$: $$z_i \sim P(z)$$. 
Then, with the prior sampled, we sample an individual data $$x_i$$ from the likelihood $$P(x | z)$$: $$x_i \sim P(x | z)$$.
Precisely, this can be represented in a graphical model below where we can see that the observed data $$x$$ is conditioned on unobserved latent variable $$z$$.

<img src = "/assets/images/VAE_graphical_model.PNG" width = "200" height = "300" class = "center">
<figcaption>Diagram showing directed graphical model for VAEs.</figcaption>
<br>
Now, remember again our goal in running inference in the VAE model is to model the latent space as good as possible given our data. This is *Bayesian Inference*,
as "inference" means calculating the posterior probability, in this case the $$P(z | x)$$. How do we calculate this? Let's look at the classic Baye's Rule: 

<p>
$$P(z | x) = \frac{P(x | z)\cdot P(z)}{P(x)}$$ 
</p>

In this case, each variable is:
<br>
$$P(z)$$ is the ***prior*** probability of $$z$$, which is the initial belief without any knowledge about $$x$$.
<br>
$$P(x)$$ is the ***evidence, or the marginal likelihood***, the probability of observing $$x$$ across all possible events.
<br>
$$P(z | x)$$ is the ***posterior*** probability of $$z$$ given $$x$$.
<br>
$$P(x | z)$$ is the ***likelihood*** of observing $$x$$ given $$z$$, which assumes the prior is correct.

From above, let's focus on the evidence, or the marginal likelihood. $$P(x)$$ can be calculated by: $$P(x) = \displaystyle \int P(x | z)P(z) dz$$ since we have a 
continuous distribution (in VAEs, the latent variable z is assumed to specified to be a Gaussian distribution with a mean of zero and unit variance ($$\mathcal{N}(0, 1)$$).
However, this simple-looking integral over the product of gaussian conditional and prior distribution is ***intractable*** because the integration is performed over 
the entire latent space, which is continuous (it is possible to have infinite number of latent variables for a single input). 

But can we try calculating $$P(x)$$ in a different way? We also know that the *joint probability* $$P(x,z) = P(x)P(z | x) $$, meaning that $$P(x) = \frac{P(x,z)}{P(z | x)}$$. 
We quickly realize that this doesn't work either since we already saw above that the posterior $$P(z | x)$$ is unknown! Therefore, we have to resort to approximating the
posterior $$P(z | x)$$ with an *approximate variational distribution $$q_\phi(z | x)$$* which has parameters $$\phi$$ that needs to be optimized. Hence, in the previous graphical
model, the dashed arrow going from x to z represents the variational approximation.

Before looking at the mathematical steps for variational approximation, let's look at VAEs in a neural network's perspective. A VAE consists of an encoder and a decoder, and both
are neural networks. The *encoder* takes in input data $$x$$ and compresses it to latent representation $$z$$, and must learn a good latent representation known as the bottleneck of the model. Note that
contrary to the encoder of the vanilla autoencoder, the encoder of the variational autoencoder will learn the mean and variance 
Therefore, the encoder can be denoted as $$p_\phi(z | x)$$, where the $$\phi$$ is the weights and biases of the model. Note that as previously mentioned, the latent space is assumed to be a Gaussian probability distribution, so sampling from the
trained encoder gets us the latent representation $$z$$ from data $$x$$. The *decoder* takes in the latent representation **z** from the encoder output and outputs the reconstructed data, or the parameters to 
the modeled probability distribution of the data space, and therefore can be denoted as $$p_\theta(x | z)$$, where $$\theta$$ is also the weights and biases. 

Note that this reconstructed probability distribution cannot be *perfect*, as the decoder learns to reconstruct the original input image only from the latent representations.
How do we ensure that the 



<img src = "/assets/images/VAE_problem.png" width = "800" height = "400" class = "center">
<figcaption>Diagram showing VAE latent space with KL-regularization (left) and without KL-regularization (right).</figcaption>

<p>
However, only having this reconstruction loss as our loss function for training the VAE is not enough. This ties back to the KL-regularization of LDMs in the previous blog (part 2),
which is the diagram showing the VAE latent space with and without KL-regularization. This is re-shown above. With an additional KL-regularization term to the VAE loss function, the "clusters" itself are bigger
and are more centered around within each other. This ensures that the decoder creates *diverse and accurate samples*, as there is smoother transitions between different classes (clusters). 
For example, for MNIST handwritten digits, if there was a cluster of 1's and a cluster of 5's, there should be a smooth transformation between 1 and 5 like they're morphing from one to another. 
However, without the KL-regularization or KL loss term in the VAE loss, we end up with small individual clusters that are far apart from each other- resulting in a latent space that is not representative 
of the data at all. Therefore, the cluster of 1's and 5's will not have a smooth transformation between one another.
</p>

<a id="model-objective"></a>
###  ***Model Objective:***

<p> 
Now why did we go over the VAEs and its variational approximation process? This is because diffusion models have a very similar set up to VAEs in
that it also has a tractable likelihood that can be maximized in a similar way. 

maximize the likelihood that an image that you generate looks like it comes from original distribution. apply same ELBO (lower bound) to the likelihood of the diffusion as well

2. **Applying Jensen's Inequality**:
   Start with the definition of the log likelihood of the data under the model:
   \[
   \log p(X) = \log \int p(X, Z) dZ
   \]
   Apply Jensen's inequality with the variational distribution \(q(Z|X)\):
   \[
   \log p(X) \geq \int q(Z|X) \log \frac{p(X, Z)}{q(Z|X)} dZ
   \]

3. **ELBO Definition**:
   Define the Evidence Lower Bound (ELBO) as the right-hand side of the inequality:
   \[
   \text{ELBO} = \int q(Z|X) \log \frac{p(X, Z)}{q(Z|X)} dZ
   \]

4. **Simplification**:
   Rearrange the terms to get a more intuitive form of ELBO:
   \[
   \text{ELBO} = \mathbb{E}_{q(Z|X)}[\log p(X|Z)] - \text{KL}(q(Z|X) || p(Z))
   \]
   where \(\text{KL}(q(Z|X) || p(Z))\) is the Kullback-Leibler divergence between the approximate posterior and the prior.

## Interpreting the ELBO Components

The ELBO can be split into two components:
- Reconstruction Loss: \(\mathbb{E}_{q(Z|X)}[\log p(X|Z)]\)
- Regularization Term: \(\text{KL}(q(Z|X) || p(Z))\)

The reconstruction loss encourages the model to generate data that resembles the input data, while the regularization term encourages the learned latent space to follow a desired prior distribution.

## ELBO Optimization

During training, our goal is to maximize the ELBO with respect to the model parameters. This involves fine-tuning the model to strike a balance between generating accurate reconstructions and maintaining a well-behaved latent space.

## Conclusion

The Evidence Lower Bound (ELBO) is a crucial concept in Variational Autoencoders (VAEs) that guides the training process by combining the reconstruction objective and regularization. Understanding the ELBO helps us grasp the underlying principles of VAEs and their role in unsupervised learning and generative modeling.

In future articles, we'll explore advanced VAE techniques and real-world applications. Stay tuned!



*Image credits to:*
- [VAE Directed Graphical Model](https://arxiv.org/pdf/1312.6114.pdf)