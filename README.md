# Hierarchical Neural Financial Forecasting
Project for Models of sequential data course at skoltech 2021

The main inspiration will be [this repository](https://github.com/KotikNikita/gluonts-hierarchical-ICML-2021) with the implementation of [End-to-End Learning of Coherent Probabilistic Forecasts for Hierarchical Time Series](http://proceedings.mlr.press/v139/rangapuram21a/rangapuram21a.pdf)
Also, an example of a high-quality project for us will be [Autoregressive Denoising Diffusion Models for Multivariate Probabilistic Time Series Forecasting](https://github.com/zalandoresearch/pytorch-ts)


# An approximate description of the operation of the algorithm
First, the time series data goes into the RNN layer to generate the distribution parameters.  
At the second stage, samples are generated from the distribution and they are projected onto the space of hierarchical.  This projected data is considered to be the predictions of the model. 
