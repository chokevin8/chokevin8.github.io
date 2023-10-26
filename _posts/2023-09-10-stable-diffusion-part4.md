---
layout: post
title:  Latent/Stable Diffusion Fully Explained! (Part 4)
date:   2023-09-10
description: Different interpretation of the training objective, explanation of training/sampling algorithm, DDIM vs DDPM sampling methods, and conditioning/classifier-free guidance!
tags: deep-learning machine-learning generative-models paper-review
categories: posts
---
---

## **Table of Contents:**
### [Background (Part 1)](/blog/2023/stable-diffusion/)
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
- ### [Summary](#summary)

---

*Note: For other parts, please click the link above in the table of contents.* 

<a id="stable-diffusion-in-numbers-2"></a>
## **Stable Diffusion In Numbers Continued**

In this last part of the blog, I want to cover the mathematical details of conditioning and also classifier-free guidance. Before, that let's briefly
look at the algorithms for training and inference, and view the training objective we derived in a different way. 

Recall equation #17 from the previous part of the blog, or the final training objective of our LDM:
<p>
$$\mathop{\arg \min}\limits_{\theta} \quad \frac{1}{2{\sigma_q}^{2}(t)} \frac{\hat{\alpha}_{t-1}(1-\alpha_t)^{2}}{(1-\hat{\alpha_t)^{2}}} [{||(\hat{x}_{\theta}(x_t,t)-x_0)||}^{2}] $$
</p>
Comparing this to equation #1 in the paper which describes the training objective, we can see that it is a bit different, as our equation above has some extra terms
and is a MSE between predicted and ground truth original image, not noise as seen below. 

*Equation #1 (Training objective) in the paper:*
<p>
$$L_{LDM} = ||\epsilon - \epsilon_{\theta}(x_t,t)||^2$$
</p>
So how are these *two somehow equivalent*? Well, let's interpret our training objective in a different way and we'll see how these two are connected. 
Recall equation #5 from the last blog (noted as equation #1 below), or the reparametrization trick we used for forward diffusion $$q(x_t \mid x_0)$$ to calculate $$x_t$$ in terms of the $$\hat{\alpha}$$s. Let's rearrange this equation in terms of
$$x_0$$ instead!
<p>
$$ x_t = \sqrt{\hat{\alpha}_t}x_0 +  \sqrt{1-\hat{\alpha}_t}{\epsilon}_0 \quad (1)$$
$$ x_0 = \frac{x_t - \sqrt{1-\hat{\alpha}_t}{\epsilon}_0}{\sqrt{\hat{\alpha}_t}} \quad (1)$$
</p>
Then, recall the equation (equation #2 below) we derived for the mean of the ground truth denoising transition step distribution $$q(x_{t-1} \mid x_t,x_0)$$, as we calculated this to minimize the KL-divergence of the ground truth and desired approximate transition step distribution to derive our training objective:

$$\mu_q = \frac{\sqrt{\alpha_t}(1-\hat{\alpha}_{t-1})x_t + \sqrt{\hat{\alpha}_{t-1}}(1-\alpha_t)x_0}{1-\hat{\alpha_t}} \quad (2)$$

Now, plug equation #1 in to $$x_0$$ of equation #2 above:
<p>
$$\mu_q = \frac{\sqrt{\alpha_t}(1-\hat{\alpha}_{t-1})x_t + \sqrt{\hat{\alpha}_{t-1}}(1-\alpha_t)\frac{x_t - \sqrt{1-\hat{\alpha}_t}{\epsilon}_0}{\sqrt{\hat{\alpha}_t}}}{1-\hat{\alpha_t}} \quad (2)$$
</p>
Skipping the rearranging algebra (you can try this yourself if you want to), we end up with:
<p>
$$\mu_q = \frac{1}{\sqrt{\alpha_t}}x_t - \frac{\sqrt{1-\alpha_t}}{\sqrt{1-\hat{\alpha_t}}\sqrt{\alpha_t}}{\epsilon}_0 \quad (3)$$
</p>
Then, like before, to find the mean of the desired approximate denoising transition distribution $$\mu_{\theta}$$, we simply replace the ground truth noise $${\epsilon}_0$$ (since we don't know ground truth distribution!) with a neural network that parametrizes 
$$\hat{\epsilon}_{\theta}(x_t,t)$$ to predict $$\epsilon_0$$ as accurately as possible to make our approximate denoising step as similar to the ground truth denoising step as possible: 
<p>
$$\mu_{\theta} = \frac{1}{\sqrt{\alpha_t}}x_t - \frac{\sqrt{1-\alpha_t}}{\sqrt{1-\hat{\alpha_t}}\sqrt{\alpha_t}}\hat{\epsilon}_{\theta}(x_t,t) \quad (4)$$
</p>
Now that equations #3 and #4 above both tell us the mean of both distributions, like what we did before, we find the KL divergence between the two. Recall the equation for calculating the
KL-divergence between two Gaussians, and plug in to find the "new" training objective:
<p>
$$ D_{KL}(\mathcal{N}(x;\mu_x,\Sigma_x) || \mathcal{N}(y;\mu_y,\Sigma_y)) = \frac{1}{2} [ \log \frac{\Sigma_y}{\Sigma_x} - d + tr({\Sigma_y}^{-1}\Sigma_x) + (\mu_y - \mu_x) ^ {T} {\Sigma_y}^{-1} (\mu_y - \mu_x) ] $$
</p>
Skipping the rearranging algebra again, we end up with our "new", different interpretation of our training objective:
<p>
$$\mathop{\arg \min}\limits_{\theta} \quad \frac{1}{2{\sigma_q}^{2}(t)} \frac{(1-\alpha_t)^{2}}{(1-\hat{\alpha_t})\alpha_t}[{||\epsilon_0 - \hat{\epsilon}_{\theta}(x_t,t)||}^{2}] \quad (5)$$
</p>
Equation #5 above is our "new" training objective, instead of predicting ground truth image, we predict the ground truth noise instead here. Empirically, depending on the use case, it may work better
to predict the noise instead of the image and vice versa, and it seems like the authors of the paper decided to predict the noise. Note that the objective function finalizes to equation #6 below because it was empirically proven that 
getting rid of the coefficient in front of the MSE term actually performed better when evaluating the performance of diffusion models. This final objective function below (equation #6) is equivalent to the loss function the authors use, which we already saw above:
<p>
$$L_{LDM} = ||\epsilon_0 - \epsilon_{\theta}(x_t,t)||^2 \quad (6)$$
</p>
Therefore, we simply end up with the mean squared error (MSE) between the ground truth noise $$\epsilon_0$$ and the predicted noise $$\epsilon_{\theta}(x_t,t)$$. 
Simply put, the decoder $$\hat{\epsilon}_{\theta}(x_t,t)$$ learns to predict the ground truth source noise $$\epsilon_0$$ that is randomly sampled from $$\mathcal{N}(0, 1)$$. The predicted source noise is the noise that originally brought the original image $$x_0$$ to the pure noised (image) $$x_t$$ via forward diffusion.
As stated in the paper, this can also be seen as a sequence of $$T$$ equally weighted autoencoders from $$t = 1,2....t-1,T$$ which predicts a denoised variant of their input $$x_t$$ in a Markov chain. As timestep reaches T, this Markovian process will then slowly converge to the ground truth input image 
$$x_0$$, assuming the training of the decoder went well. 

<a id="training-inference"></a>
###  ***Training and Inference:***

Now that we've derived the training (loss) objective from scratch, let's briefly go over the entire training and the inference algorithm:

<img src = "/assets/images/train_inference_algorithm.png" width = "985" height = "250" class = "center">
<figcaption>The training and inference algorithm, summarized.</figcaption>
<br>
Let's first look at the training algorithm:
1. We repeat the below process (steps 2~5) until convergence or a preset number of epochs. 
2. Sample an image $$x_0$$ from our dataset/data distribution, $$q(x_0)$$.
3. Sample t, or timestep from 1 to $$T$$.
4. Sample noise from a normal distribution $$\epsilon \sim \mathcal{N}(0, I)$$
5. Take gradient descent step on the training objective we just derived $$L_{LDM} = ||\epsilon - \epsilon_{\theta}(x_t,t)||^2 $$ with respect to $$\theta$$, which is the 
parameters of the weights and biases of the decoder.

Not too bad! What about the above sampling algorithm? 

Before describing the sampling process, let's look back at the training objective, specifically regarding the value of T, or the timesteps for the forward process. 
Above, we described this as a Markovian process, and ideally we would like to *maximize* $$T$$ or the number of timesteps in the forward diffusion so that the reverse process
can be as close to a Gaussian as possible so that the generative process is accurate and generates a good image quality. 

However, in this Markovian sampling process called DDPM (short for Denoising Diffusion Probabilistic Models), the $$T$$ timesteps have to be performed sequentially, meaning
sampling speed is extremely slow, especially compared to fast sampling speeds of predecessors such as GANs. The above sampling algorithm is the DDPM sampling algorithm, which will be explained first,
but then we will also mention a new, non-Markovian sampling process called DDIM (short for Denoising Diffusion Implicit Models) that is able to accelerate sampling speeds. **Note that the authors of the
LDM paper utilized DDIM because of this exact reason.**

Remember that for sampling, we only need the trained decoder from above (no encoder). Therefore, we sample latent noise $$x_T$$ from prior $$p(x_T)$$, which is $$\epsilon \sim \mathcal{N}(0, I)$$
and then run the series of $$T$$ equally weighted autoencoders as mentioned before in a Markovian style. The equation shown in the sampling algorithm is essentially identical to equation #4 above, or:
<p>
$$ \mu_{\theta} = \frac{1}{\sqrt{\alpha_t}}x_t - \frac{\sqrt{1-\alpha_t}}{\sqrt{1-\hat{\alpha_t}}\sqrt{\alpha_t}}\hat{\epsilon}_{\theta}(x_t,t) $$
</p>

The equation in the sampling algorithm just has an additional noise term $$\sigma_tz$$ for stochasticity during sampling. Assuming our training went well, we now have the neural network $$\hat{\epsilon}_{\theta}(x_t,t)$$ trained that predicts the noise $$\epsilon$$ for given input image $$x_t$$. Now, remember that this neural network
in our LDM is our U-Net architecture with the attention layers that predicts the noise given our input image. Inputting a timestep $$t$$ and original image $$x_t$$ to the trained neural network gives us the predicted noise $$\epsilon$$, and using that we can sample $$x_{t-1}$$ until $$t=1$$. When $$t=1$$, we have
our sampled output of $$x_0$$. However, as discussed above, Denoising Diffusion Implicit Model (DDIM) uses a non-Markovian sampling process that makes the process much quicker. Essentially, DDIM uses $$S$$ steps instead of $$T$$ where $$S<T$$, and the authors of the LDM paper
therefore use *DDIM over DDPM.*

To derive the DDIM sampling process, we utilize the *reparametrization trick* again, which we applied previously for forward diffusion. 
We can use $$ x = \mu + \sigma * \epsilon$$ to essentially alter our sampling process $$q(x_{t-1}|x_t,x_0)$$ to be parametrized by another random variable,
a desired standard deviation $$\epsilon_t$$. The reparametrization is shown below:
<p>
$$\text{Recall equation #1:} \, x_t = \sqrt{\hat{\alpha}_t}x_0 +  \sqrt{1-\hat{\alpha}_t}{\epsilon}_0 $$
$$\text{The equation for} \quad x_{t-1} \quad \text{instead is:} \quad x_{t-1} = \sqrt{\hat{\alpha}_{t-1}}x_0 + \sqrt{1-\hat{\alpha}_{t-1}}{\epsilon}_{t-1} $$
$$\text{Add extra term} \quad \sigma_t \epsilon \quad \text{where} \quad \sigma_t^{2} \quad \text{is the variance of our distribution}: \quad x_{t-1} \quad \text{instead is:} \quad x_{t-1} = \sqrt{\hat{\alpha}_{t-1}}x_0 + \sqrt{1-\hat{\alpha}_{t-1}-\sigma_t^{2}}\epsilon_t + \sigma_t \epsilon $$
$$\text{Since} \quad epsilon_t = \frac{x_t - \sqrt{\hat{\alpha_t}}x_0}{\sqrt(1-\hat{\alpha_t}}:$$
$$ x_{t-1} = \sqrt{\hat{\alpha}_{t-1}}x_0 + \sqrt{1-\hat{\alpha}_{t-1}-\sigma_t^{2}}\sqrt{\hat{\alpha_t}}x_0}{\sqrt(1-\hat{\alpha_t}}  + \sigma_t \epsilon
</p>

The main advantages of DDIM over DDPM are:
stochastic, allows consistency and also interpolation (interpolation in DDPM is possible, but stochasticity ruins it)
1. Consistency: DDIMs are consistent, meaning that if we initialize the same latent variable $$x_T$$ via same random seed during sampling, the samples 
2. 

<a id="conditioning"></a>
###  ***Conditioning:***
For conditioning, look at table 15 of LDM paper
maybe include autoencoder training as well since conditioning is basically UNet training details + pretrained encoder.
Autoencoder training is in appendix G: Details on Autoencoder Models

<a id="classifier-free-guidance"></a>
###  ***Classifier-Free Guidance:***

<a id="summary"></a>
###  ***Summary:***
