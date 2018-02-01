# Gae In Pytorch
Graph Auto-Encoder in PyTorch

This is a PyTorch/Pyro implementation of the Variational Graph Auto-Encoder model described in the paper:
 
T. N. Kipf, M. Welling, [Variational Graph Auto-Encoders](https://arxiv.org/abs/1611.07308), NIPS Workshop on Bayesian Deep Learning (2016)

This repository uses some of the code found here: https://github.com/tkipf/pygcn and https://github.com/tkipf/gae. 

### Requirements
- Python 2.7
- Pyro
- PyTorch
- networkx
- scikit-learn
- scipy
- numpy
- matplotlib
- pickle


### To run
After installing all requirements:
```bash
python train.py
```

### Notes
- This implementation uses Pyro's blackbox SVI function with the default ELBO loss. This is slower than the TensorFlow implementation which uses a custom loss function with an analytic solution to the KL divergence term. 
- Currently the code is not set up to use a GPU, but the code should be easy to extend to improve running speed
