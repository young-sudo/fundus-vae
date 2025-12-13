# Glaucoma Detector with VAE

*by Younginn Park*

![Python](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white)
![Kaggle](https://img.shields.io/badge/Kaggle-20BEFF?style=for-the-badge&logo=Kaggle&logoColor=white)
![Jupyter](https://img.shields.io/badge/Jupyter-F37626?style=for-the-badge&logo=jupyter&logoColor=white)
![TensorFlow](https://img.shields.io/badge/TensorFlow-FF6F00?style=for-the-badge&logo=tensorflow&logoColor=white)
![NumPy](https://img.shields.io/badge/NumPy-013243?style=for-the-badge&logo=numpy&logoColor=white)
![Seaborn](https://img.shields.io/badge/Seaborn-82C8D9?style=for-the-badge&logo=chartdotjs&logoColor=black) 

This project presents a Variational Autoencoder (VAE) implemented in TensorFlow. Originally developed in a **Kaggle notebook**, the model is applied to the **SMDG-19 Fundus Glaucoma Detection dataset**, a comprehensive collection that merges 19 public datasets into a single standardized format, to explore unsupervised learning techniques for medical imaging. The project demonstrates the use of VAEs for dimensionality reduction, feature extraction, and potential glaucoma-related anomaly detection in retinal images.

<p align="center">
  <img src="https://raw.githubusercontent.com/young-sudo/fundus-vae/main/img/fundus.png" alt="fundus" width="600"/>
</p>

Originally written in Kaggle notebook for the [Fundus Glaucoma Detection Dataset](https://www.kaggle.com/datasets/sabari50312/fundus-pytorch).

# Methods

## Vanilla VAE

Variational Autoencoders (VAEs) are a class of generative models that learn a probabilistic mapping between observed data $x$ and a set of latent variables $z$. A standard VAE consists of an encoder that approximates the posterior distribution $q(z|x)$ and a decoder that defines the likelihood $p(x|z)$, with a simple prior $p(z)$, typically a standard normal distribution. Training is performed by maximizing the Evidence Lower Bound (ELBO), which balances accurate reconstruction of the data with regularization of the latent space via a Kullback–Leibler divergence term, encouraging meaningful and continuous latent representations.

* Probabilistic model: $p(x,z)=p(x|z)p(z)$
* Inference model: $q(z|x)$
* ELBO formulation: $\mathbb{E}{q(z|x)}[\log p(x|z)]-D{KL}(q(z|x),|,p(z))$

<div align="center">
  <figure>
    <img src="https://raw.githubusercontent.com/young-sudo/fundus-vae/main/img/vae.png" alt="tests" width="500"/><br>
    <figcaption style="text-align:center;"><em>Architecture of a Vanilla VAE model (source: medium)</em></figcaption>
  </figure>
</div>

## Hierarchical VAE

Hierarchical Variational Autoencoders (VAEs) represent an advanced evolution of the standard VAE framework, featuring a multi-layered latent variable structure. This hierarchical structure, akin to a Bayesian network, enables capturing complex data distributions through a nuanced representation with multiple latent layers, where each layer is dependent on its predecessor.

In a simple hierarchical VAE model with two latent layers, $z_2$ and $z_1$, the relationship is such that $z_1$ is dependent on $z_2$, forming a sequential structure $z_2 \rightarrow z_1$. This leads to a distinct alteration in both the VAE's architecture and its probabilistic model:
- **Probabilistic model**: $p(x,z_1,z_2)=p(x|z_1)p(z_1|z_2)p(z_2)$
- **Inference model**: $q(z_1,z_2|x)=q(z_1|x)q(z_2|z_1)$
- **ELBO formulation**: $E_{q(z_1|z_2)}-D_{KL}(q(z_1|x)‖p(z_1))-D_{KL}(q(z_2|z_1)‖p(z_2))$

## Implementation

3 VAE architectures were taken into consideration:
- Simple VAE - having simple Encoder and Decoder with 3 convolutional layers (dims=16,32,64) and 1 latent layer
- Hierarchical VAE - with the Encoder and Decoder like in the Simple VAE, but 2 latent layers
- AlexNet-like VAE - with the Encoder and Decoder architecture that mimics the one used in 2012 AlexNet with 1 latent layer ([Wikipedia](https://en.wikipedia.org/wiki/AlexNet))

Ultimately, the Simple VAE architecture was chosen for the best performance.

Different sets of parameters were tested:

<div align="center">
  <figure>
    <img src="https://raw.githubusercontent.com/young-sudo/fundus-vae/main/img/tests.png" alt="tests" width="500"/><br>
    <figcaption style="text-align:center;"><em>Tests for parameters beta (strength of KL Loss) and latent dimensions</em></figcaption>
  </figure>
</div>

The final architecture of the model looked like this:

<div align="center">
  <figure>
    <img src="https://raw.githubusercontent.com/young-sudo/fundus-vae/main/img/model.png" alt="fundus" width="500"/><br>
    <figcaption style="text-align:center;"><em>Architecture of the VAE model.</em></figcaption>
  </figure>
</div>
