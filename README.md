# Glaucoma Detector using VAE

*by Younginn Park*

![Python](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white)
![Jupyter](https://img.shields.io/badge/Jupyter-F37626?style=for-the-badge&logo=jupyter&logoColor=white)
![TensorFlow](https://img.shields.io/badge/TensorFlow-FF6F00?style=for-the-badge&logo=tensorflow&logoColor=white)
![NumPy](https://img.shields.io/badge/NumPy-013243?style=for-the-badge&logo=numpy&logoColor=white)
![Seaborn](https://img.shields.io/badge/Seaborn-82C8D9?style=for-the-badge&logo=chartdotjs&logoColor=black)

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
    <img src="img/model.png" alt="fundus" width="500"/><br>
    <figcaption style="text-align:center;"><em>Figure 1. Architecture of the VAE model.</em></figcaption>
  </figure>
</div>
