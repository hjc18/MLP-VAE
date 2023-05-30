# MLP-VAE
MLP-based VAE model on MNIST dataset, implemented with PyTorch.

- Both the Bernoulli MLP and the Gaussian MLP are used as the decoder. (see [the VAE paper](https://arxiv.org/abs/1312.6114))
- The encoder is considered to be a Gaussian MLP.
- Here are some random samples from the Bernoulli VAE.
![bernoulli](./figure/bernoulli.png)
- and some random samples from the Gaussian VAE.
![gaussian](./figure/gaussian.png)