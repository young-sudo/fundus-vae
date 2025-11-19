# Glaucoma Detector using VAE

*by Younginn Park*

This project presents a Variational Autoencoder (VAE) implemented in TensorFlow. Originally developed in a **Kaggle notebook**, the model is applied to the **Fundus Glaucoma Detection dataset** to explore unsupervised learning techniques for medical imaging. The project demonstrates the use of VAEs for dimensionality reduction, feature extraction, and potential glaucoma-related anomaly detection in retinal images.

<p align="center">
  <img src="img/fundus.png" alt="fundus" width="600"/>
</p>

Originally written in Kaggle notebook for the [Fundus Glaucoma Detection Dataset](https://www.kaggle.com/datasets/sabari50312/fundus-pytorch).

3 VAE architectures were taken into consideration:
- Simple VAE - having simple Encoder and Decoder with 3 convolutional layers (dims=16,32,64) and 1 latent layer
- Hierarchical VAE - with the Encoder and Decoder like in the Simple VAE, but 2 latent layers
- AlexNet-like VAE - with the Encoder and Decoder architecture that mimics the one used in 2012 AlexNet with 1 latent layer ([Wikipedia](https://en.wikipedia.org/wiki/AlexNet))

Ultimately, the Simple VAE architecture was chosen for the best performance.

<div align="center">
  <figure>
    <img src="img/model.png" alt="fundus" width="700"/>
    <figcaption style="text-align:center;"><em>Figure 1. Architecture of the VAE model.</em></figcaption>
  </figure>
</div>
