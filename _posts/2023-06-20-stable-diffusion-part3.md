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
### [Background](#background) ([Part 1](/blog/2023/stable-diffusion/))
- ###  [Introduction](#introduction)
- ### [Stable Diffusion vs GAN](#stable-diffusion-vs-gan)

### [Stable Diffusion](#stable-diffusion) ([Part 2](/blog/2023/stable-diffusion-part2/))
- ### [Motivation](#motivation)
- ### [Model Architecture](#model-architecture)
- ### [Experiments & Results](#experiment-results)

### [Math and Details Behind Stable Diffusion](#math-behind-stable-diffusion) (Part 3- This Blog!)
- ### [Background](#background)
- ### [Model Objective](#model-objective)
- ### [Autoencoder](#autoencoder)
- ### [U-Net](#u-net)
- ### [Pretrained Encoder](#pretrained-encoder)

---

*Note: For Part 1 and 2, please click the link above in the table of contents.* 

<a id="Background"></a>
##  **Background:**

In this last part of the blog, I will cover most of the important mathematical details behind latent diffusion that is necessary to fully understand
how latent diffusion works. Before looking at the model objective, I think it's important to do a quick review of the background:

Let's look at variational autoencoders (VAEs) in a probabilistic way. The variational autoencoder holds a probability model with the $x$ representing
the data, and the $z$ representing the latent variables of the autoencoder. Remember that we want our latent variable $$z$$ to model the data $$x$$ as 
accurately as possible. Note that $$x$$ can be seen, but $$z$$ cannot since it is in the latent space. To perform the generative process, or run inference, 
for each individual data $$j$$, we first sample latent variable $$z_i$$ from the prior $$P(z)$$: $$z_i \sim P(z)$$. 
Then, with the prior sampled, we sample an individual data $$x_i$$ from the likelihood $$P(x | z)$$: $$x_i \sim P(x | z)$$.
Precisely, this can be represented in a graphical model below where we can see that the observed data $$x$$ is conditioned on unobserved latent variable $$z$$.

<img src = "/assets/images/VAE_graphical_model.PNG" width = "200" height = "300" class = "center">
<figcaption>Diagram showing directed graphical model for VAEs.</figcaption>

Now, remember again our goal in running inference in the VAE model is to model the latent space as good as possible given our data. This is *Bayesian Inference*,
as "inference" means calculating the posterior probability, in this case the $$P(z | x)$$. How do we calculate this? Let's look at the classic Baye's Rule: 

<p>
$$P(z | x) = \frac{P(x | z)\cdot P(z)}{P(x)}$$ 
</p>

In this case, each variable is:
$$P(z)$$ is the prior probability of $$z$$, which is the initial belief without any knowledge about $$x$$.
$$P(x)$$ is the evidence, or the marginal likelihood, the probability of observing $$x$$ across all possible events.
$$P(z | x)$$ is the posterior probability of $$z$$ given $$x$$.
$$P(x | z)$$ is the likelihood of observing $$x$$ given $$z$$, which assumes the prior is correct.

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
Therefore, the encoder can be denoted as $$q_\phi(z | x)$$, where the $$\phi$$ is the weights and biases of the model. Note that as previously mentioned, the latent space is assumed to be a Gaussian probability distribution, so sampling from the
trained encoder gets us the latent representation $$z$$ from data $$x$$. The *decoder* takes in the latent representation **z** from the encoder output and outputs the reconstructed data, or the parameters to 
the modeled probability distribution of the data space, and therefore can be denoted as $$p_\theta(x | z)$$, where $$\theta$$ is also the weights and biases. 

Note that this reconstructed probability distribution cannot be *perfect*, as the decoder learns to reconstruct the original input image only from the latent representations.
However, we can optimize a loss function during our training to minimize this reconstruction loss, which is simple as a 


<a id="model-objective"></a>
##  **Model Objective:**
<p> 
Now why did we go over the basics of VAEs and variational approximation process? This is because diffusion models have a very similar set up to VAEs in
that it also has a tractable likelihood that can be maximized in a similar way. 


# Understanding the Evidence Lower Bound (ELBO) in Variational Autoencoders (VAEs)

Variational Autoencoders (VAEs) are powerful generative models that allow us to learn meaningful representations of data. One key concept in VAEs is the Evidence Lower Bound (ELBO), which plays a crucial role in the training process and guides us in optimizing the model's parameters. In this article, we'll break down the ELBO and its significance step by step.

## Introduction to VAEs

Before diving into ELBO, let's briefly understand what VAEs are. VAEs are a type of generative model that combine the strengths of autoencoders and probabilistic modeling. They allow us to learn a latent representation of data that can be used for tasks like data generation, interpolation, and more.

## The ELBO Concept

The Evidence Lower Bound (ELBO) is a fundamental equation in the context of VAEs. It helps us formulate the training objective and guide the optimization process. The ELBO is derived from the idea of maximizing the marginal likelihood of the data.

## Deriving the ELBO

Let's walk through the derivation of the ELBO step by step:

1. **Assumptions**:
   - We have observed data \(X\) and latent variables \(Z\).
   - \(p(X, Z)\) is the joint distribution of data and latent variables.
   - \(q(Z|X)\) is the approximate posterior distribution over latent variables.

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


# Exploring Stochastic Gradient Langevin Dynamics (SGLD)

Stochastic Gradient Langevin Dynamics (SGLD) is a powerful optimization algorithm that combines the concepts of stochastic gradient descent and Langevin dynamics. It's widely used in machine learning and Bayesian inference to explore complex high-dimensional spaces efficiently. In this article, we'll dive into the workings of SGLD and understand how it can be applied to various scenarios.

## Introduction to SGLD

SGLD is a variant of the Langevin Monte Carlo method, which originates from statistical physics. It's designed to sample from complex probability distributions by introducing noise into the gradient updates. This noise mimics the randomness seen in real-world data and helps the algorithm escape local minima and explore the entire distribution.

## Basics of Langevin Dynamics

Langevin Dynamics is a concept borrowed from physics, where particles follow a trajectory influenced by both a deterministic force and random noise. In the context of optimization, Langevin Dynamics can be used to sample from a target distribution by iteratively updating the position of a point.

## Incorporating Stochastic Gradient Descent

SGLD combines Langevin Dynamics with Stochastic Gradient Descent (SGD) to create a practical optimization algorithm for machine learning. In each step of the algorithm, SGLD updates the model parameters using both the negative log-likelihood gradient (SGD component) and a random noise term (Langevin Dynamics component).

## SGLD Algorithm

Here's the basic outline of the SGLD algorithm:

1. **Initialization**: Start with initial parameters \(\theta_0\).

2. **Iteration**:
   - Sample a mini-batch of data \(D\) from the dataset.
   - Calculate the gradient of the negative log-likelihood: \(\nabla \mathcal{L}(\theta | D)\).
   - Introduce noise to the gradient using random noise: \(\eta \sim \mathcal{N}(0, \epsilon I)\).
   - Update the parameters using the SGLD update rule:
     \[
     \theta_{t+1} = \theta_t - \frac{\epsilon}{2} \nabla \mathcal{L}(\theta_t | D) + \eta
     \]

3. **Repeat Iteration**: Perform multiple iterations to optimize the parameters.

## Advantages and Challenges

SGLD offers several advantages:
- It efficiently explores complex high-dimensional spaces.
- It can escape local minima and avoid getting stuck.
- It incorporates data-driven noise, making it suitable for noisy data.

However, SGLD also has challenges:
- Setting the noise level (\(\epsilon\)) is crucial; too high or too low values can impact convergence.
- The algorithm might converge slowly for certain distributions.

## Applications

SGLD finds applications in various fields:
- Bayesian Inference: Sampling from posterior distributions.
- Neural Network Training: Optimizing deep neural networks.
- Generative Models: Training generative adversarial networks (GANs) and variational autoencoders (VAEs).

## Conclusion

Stochastic Gradient Langevin Dynamics (SGLD) bridges the gap between optimization and probabilistic modeling. By combining Langevin Dynamics with stochastic gradient descent, SGLD efficiently explores complex probability distributions, making it a valuable tool in machine learning and Bayesian inference. Understanding SGLD opens the door to exploring high-dimensional spaces and training advanced models that handle uncertainty effectively.

In future articles, we'll dive deeper into SGLD's applications and advanced techniques. Stay curious!

*Image credits to:*
- [VAE Directed Graphical Model](https://arxiv.org/pdf/1312.6114.pdf)