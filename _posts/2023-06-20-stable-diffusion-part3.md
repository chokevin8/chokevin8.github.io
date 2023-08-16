---
layout: post
title:  Latent/Stable Diffusion for Beginners! (Part 3)
date:   2023-06-20
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

### [Stable Diffusion In Numbers (Part 3)](#stable-diffusion-in-numbers-1) (This Blog!)
- ### [VAEs and ELBO](#vaes-elbo)
- ### [Model Objective](#model-objective)

### [Stable Diffusion In Numbers Continued (Part 4)](/blog/2023/stable-diffusion-part4/)
- ### Conditioning 
- ### Classifier-Free Guidance

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

<img src = "/assets/images/VAE_graphical_model.PNG" width = "400" height = "420" class = "center">
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

But *how do we ensure that our approximate variational distribution $$q_\phi(z | x)$$ will be as similar as possible to the intractable posterior $$P(z | x)$$?*
We do this by minimizing the KL-divergence between the two distributions. For two distributions P and Q, KL-divergence essentially measures the difference between the two distributions.
The value of the KL-divergence cannot be less than zero, as zero denotes that the two distributions are perfectly equal to each other. Note that $$D_{KL}(P || Q) = \sum_{n=i} P(i) \log \frac{P(i)}{Q(i)} $$ 
<br>
Now let's expand on this:

<p>
$$min(D_{KL}(q(z|x) || P(z|x))) = - \sum_{n=i} q(z|x) \log \frac{P(z|x)}{q(z|x)} \ (1)$$ 
$$min(D_{KL}(q(z|x) || P(z|x))) = - \sum_{n=i} q(z|x) \log \frac{P(x,z)}{q(z|x)P(x)} \ (2) \quad \text{since} \ P(z|x) = \frac {P(x,z)}{P(x)}$$
$$min(D_{KL}(q(z|x) || P(z|x))) = - \sum_{n=i} q(z|x) [\log \frac{P(x,z)}{q(z|x)} - \log P(x)] \ (3)$$
$$min(D_{KL}(q(z|x) || P(z|x))) = - \sum_{n=i} q(z|x) \log \frac{P(x,z)}{q(z|x)} + \sum_{n=i} q(z|x) \log P(x) \ (4)$$
$$min(D_{KL}(q(z|x) || P(z|x))) = - \sum_{n=i} q(z|x) \log \frac{P(x,z)}{q(z|x)} +  \log P(x) \ (5) \quad \text{since} \ \sum_{n=i} q(z|x) = 1$$
$$\log P(x) = min(D_{KL}(q(z|x) || P(z|x))) + \sum_{n=i} q(z|x) \log \frac{P(x,z)}{q(z|x)} \ (6)$$
</p>
Observe the left hand side of the last line above (or equation #6). Since we observe $$x$$, this is simply a constant. Observe the right hand side now. 
The first term is what we initially wanted to minimize. The second term is what's called an ***"evidence/variational lower bound", or "ELBO" for short***.
This is because rather than minimizing our first term (KL-divergence), utilizing the fact that the KL-divergence is always greater than or equal to zero, we can
rather maximize the ELBO instead. We know understand why the second term is called a "lower bound", as $$ELBO \leq \log P(x)$$ since $$D_{KL}(q(z|x) || P(z|x)) \geq 0$$ 
all the time. 
<br>
Now let's expand on ELBO:
<p>
$$ ELBO = \sum_{n=i} q(z|x) \log \frac{P(x,z)}{q(z|x)} \ (1)$$ 
$$ ELBO = \sum_{n=i} q(z|x) \log \frac{P(x|z)P(z)}{q(z|x)} \ (2)$$
$$ ELBO = \sum_{n=i} q(z|x) [\log P(x|z) + \log \frac{P(z)}{q(z|x)}] \ (3)$$
$$ ELBO = \sum_{n=i} q(z|x) \log P(x|z) + \sum_{n=i} q(z|x) \log \frac{P(z)}{q(z|x)} \ (4)$$
$$ ELBO = \sum_{n=i} q(z|x) \log P(x|z) - D_{KL}(q(z|x) || P(z)) \ (5)$$ 
$$ ELBO = \mathbb{E}_{q(z|x)} [\log P(x|z)] - D_{KL}(q(z|x) || P(z)) \ (6)$$ 
</p>

***Remember*** the last line above (or equation #6) for later, this is also the ***loss function*** for training VAEs. But to understand this expression better, let's now look at VAEs in a *neural network's perspective*. A VAE consists of an encoder and a decoder, and both
are neural networks. The *encoder* takes in input data $$x$$ and compresses it to latent representation $$z$$, and must learn a good latent representation known as the bottleneck of the model. Note that
contrary to the encoder of the vanilla autoencoder, the encoder of the variational autoencoder will learn the mean and variance 
Therefore, the encoder can be denoted as $$q_\phi(z | x)$$, where the $$\phi$$ is the weights and biases of the model. Note that as previously mentioned, the latent space is assumed to be a Gaussian probability distribution, so sampling from the
trained encoder gets us the latent representation $$z$$ from data $$x$$. The *decoder* takes in the latent representation **z** from the encoder output and outputs the reconstructed data denoted as $$\hat{x}$$, or the parameters to 
the modeled probability distribution of the data space, and therefore can be denoted as $$p_\theta(x | z)$$, where $$\theta$$ is also the weights and biases. The below diagram helps us see this entire scheme.

<img src = "/assets/images/autoencoder_diagram.png" width = "800" height = "420" class = "center">
<figcaption>Diagram showing autoencoder architecture.</figcaption>
<br>
Now let's go back to the remembered equation that I just mentioned. Let's look at the first term $$\mathbb{E}_{q(z|x)} [\log P(x|z)]$$. Now, remember that the latent space $$z$$ is assumed to be a
Gaussian distribution $$z_i \sim \mathcal{N}(0,1)$$. Observe this:
<p>
$$\log p(x|z) \sim \log exp(-(x-f(z))^2$$
$$\sim |x-f(z)|^2 $$
$$ = |x-\hat{x}|^2 $$
</p>
where $$f(z) = \hat{x}$$, as the reconstructed image $$\hat{x}$$ is the distribution mean $$f(z)$$. This is because $$ p(x|z) \sim \mathcal{N}(f(z), I)$$. Therefore, here we see that
the first term is correlated to the mean squared error (MSE) loss between the original image and the reconstructed image. This makes sense, as during training, we want to make penalize the model
if the reconstructed image is too dissimilar to the original image. It is important to see that this was the first term of the *ELBO*, and remember we want to maximize this. Maximizing the first term
is then therefore correlated to minimizing the MSE/reconstruction loss.

Let's now look at the second term, $$-D_{KL}(q(z|x) || P(z))$$ (note the negative sign) which is the KL-divergence between our learned gaussian distribution (encoder) $$q(z|x)$$ 
and the prior (latent space) gaussian distribution. Remember this is the second term of ELBO, so we still want to maximize- but note the negative sign, we
actually want to minimize the KL divergence between the two- which makes sense as we want to encourage the learned distribution from the encoder to be similar to the unit Gaussian prior.

<img src = "/assets/images/VAE_problem.png" width = "800" height = "400" class = "center">
<figcaption>Diagram showing VAE latent space with KL-regularization (left) and without KL-regularization (right).</figcaption>
<br>

This actually ties back to the KL-regularization of LDMs in the previous blog (part 2), which is the diagram showing the VAE latent space with and without KL-regularization. This is re-shown above. 
The minimization of KL divergence shown above regularizes the latent space as the "clusters" itself are bigger and are more centered around within each other. This ensures that the decoder creates <i>diverse and accurate samples</i>, as there 
would be smoother transitions between different classes (clusters). This is why both reconstruction loss term and KL-divergence term are included in the VAE loss function during training.

<img src = "/assets/images/mnist_latent_space.jpg" width = "600" height = "600" class = "center">
<figcaption>Diagram showing regularized VAE latent space of MNIST dataset.</figcaption>
<br>
For example, as seen above for MNIST handwritten digits, we see that the classes, or clusters have a smooth transition in this latent space. Without regularization, the encoder can cheat by learning narrow distributions
with low variances. Now that we've understood the importance of maximizing ELBO to train a VAE, let's go back to LDMs.

---

<a id="model-objective"></a>
###  ***Model Objective:***

Now why did we go over the VAEs and its variational approximation process? This is because diffusion models have a very similar set up to VAEs in
that it also has a tractable likelihood that can be maximized by maximizing the ELBO to ensure that the approximate posterior is as similar as possible
to the unknown true posterior we'd like to model. We're going to derive the training loss function, or the *model objective* just like
how it was done for VAEs. 

Let's first look at the forward and the backward diffusion process in a probabilistic way, since we already know about the diffusion processes in neural networks (train
a regularized autoencoder for forward and backward diffusion process!). Take a look at the diagram below:

<img src = "/assets/images/forward_backward_diffusion.png" width = "1325" height = "258" class = "center">
<figcaption>Diagram showing the forward and reverse diffusion process.</figcaption>
<br>
The forward diffusion process actually is the reverse of the above diagram, as the arrows should be going the opposite way- the forward diffusion process adds noise to a specific
data point $$x_0$$ that is sampled from the unknown, true distribution we'd like to model. Then, $$x_0$$ has Gaussian noise added to it in a Markovian process (from $$x_{t-1}$$ all the way to $$x_T$$) with $$T$$ steps.
Therefore, $$q(x_t|x_{t-1})$$ takes the image and outputs a slightly more noisy version of the image. This can be formulated below:
<p>
$$q(x_t|x_{t-1}) = \mathcal{N}(x_t; \mu_t = \sqrt{1-\beta_t}x_{t-1},\Sigma_t = \beta_t = \beta_tI)$$
</p>
*Note that above process can be made non-Markovian in a different sampling process called DDIM(remember in Part 2, I mentioned diffusion process is either Markovian or non-Markovian, this is DDPM vs DDIM, this will be explained
in next part of this blog).*

Assuming high-dimensionality, $$q(x_t|x_{t-1})$$ is a Gaussian distribution with the above defined mean and variance. Note that for each dimension, it has the same standard deviation $$\beta_t$$.
$$\beta_t$$ is a number between 0 and 1, and essentially scales the data so the variance doesn't grow out of proportion. The authors use a *linear schedule* for $$\beta_t$$, meaning that $$\beta_t$$ is linearly
increased as the image gets noised more. Note that with above formula, we can easily obtain desired noised image at timestep $$T$$ by using the Markovian nature of the process. Below is a tractable, closed-form 
formula to sample a noised image at any timestep:
<p>
$$q(x_{1:T}|x_0) = \prod_{t=1}^{T} q(x_t|x_{t-1})
</p>
Basically, if T = 200 timesteps, we would have 200 products to sample the noised image $$x_{t=200}$$. However, if the timestep gets larger, we run in to trouble of computational issues. Therefore,
we utilize the *reparametrization trick* which gives us a much simpler tractable, closed-form formula for sampling that requires much less computations:

<p>

</p>

After deriving training objective:
LDM use DDIM, while Markvovian above is DDPM. Note training objective is the same. Short detail on DDIM: 



maximize the likelihood that an image that you generate looks like it comes from original distribution. apply same ELBO (lower bound) to the likelihood of the diffusion as well



---

The next part (last part of blog on stable diffusion) will cover more mathematical details on LDMs that were not covered in this part of the blog.

*Image credits to:*
- [VAE Directed Graphical Model](https://arxiv.org/pdf/1312.6114.pdf)
- [MNIST Latent Space Example](https://www.tensorflow.org/tutorials/generative/cvae)