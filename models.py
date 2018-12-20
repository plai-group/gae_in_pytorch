import torch
import torch.nn as nn
import torch.nn.functional as F
import pyro
import pyro.distributions as dist

from layers import GraphConvolution
from utils import get_subsampler
from dist import WeightedBernoulli
pyro.enable_validation(True)

class GCNEncoder(nn.Module):
    """Encoder using GCN layers"""

    def __init__(self, n_feat, n_hid, n_latent, dropout):
        super(GCNEncoder, self).__init__()
        self.gc1 = GraphConvolution(n_feat, n_hid)
        self.gc2_mu = GraphConvolution(n_hid, n_latent)
        self.gc2_sig = GraphConvolution(n_hid, n_latent)
        self.dropout = dropout


    def forward(self, x, adj):
        # First layer shared between mu/sig layers
        x = F.relu(self.gc1(x, adj))
        x = F.dropout(x, self.dropout, training=self.training)
        mu = self.gc2_mu(x, adj)
        log_sig = self.gc2_sig(x, adj)
        return mu, torch.exp(log_sig)


class InnerProductDecoder(nn.Module):
    """Decoder for using inner product for prediction."""

    def __init__(self, dropout):
        super(InnerProductDecoder, self).__init__()
        self.dropout = dropout
        self.sigmoid = nn.Sigmoid()
        self.fudge = 1e-7


    def forward(self, z):
        z = F.dropout(z, self.dropout, training=self.training)
        adj = (self.sigmoid(torch.mm(z, z.t())) + self.fudge) * (1 - 2 * self.fudge)
        return adj


class GAE(nn.Module):
    """Graph Auto Encoder (see: https://arxiv.org/abs/1611.07308)"""

    def __init__(self, data, n_hidden, n_latent, dropout, subsampling=False):
        super(GAE, self).__init__()

        # Data
        self.x = data['features']
        self.adj_norm = data['adj_norm']
        self.adj_labels = data['adj_labels']
        self.obs = self.adj_labels.view(1, -1)

        # Dimensions
        N, D = data['features'].shape
        self.n_samples = N
        self.n_edges = self.adj_labels.sum()
        self.n_subsample = 2 * self.n_edges
        self.input_dim = D
        self.n_hidden = n_hidden
        self.n_latent = n_latent

        # Parameters
        self.pos_weight = float(N * N - self.n_edges) / self.n_edges
        self.norm = float(N * N) / ((N * N - self.n_edges) * 2)
        self.subsampling = subsampling

        # Layers
        self.dropout = dropout
        self.encoder = GCNEncoder(self.input_dim, self.n_hidden, self.n_latent, self.dropout)
        self.decoder = InnerProductDecoder(self.dropout)


    def model(self):
        # register PyTorch module `decoder` with Pyro
        pyro.module("decoder", self.decoder)

        # Setup hyperparameters for prior p(z)
        z_mu    = torch.zeros([self.n_samples, self.n_latent])
        z_sigma = torch.ones([self.n_samples, self.n_latent])

        # sample from prior
        z = pyro.sample("latent", dist.Normal(z_mu, z_sigma).to_event(2))

        # decode the latent code z
        z_adj = self.decoder(z).view(1, -1)

        # Score against data
        pyro.sample('obs', WeightedBernoulli(z_adj, weight=self.pos_weight).to_event(2), obs=self.obs)


    def guide(self):
        # register PyTorch model 'encoder' w/ pyro
        pyro.module("encoder", self.encoder)

        # Use the encoder to get the parameters use to define q(z|x)
        z_mu, z_sigma = self.encoder(self.x, self.adj_norm)

        # Sample the latent code z
        pyro.sample("latent", dist.Normal(z_mu, z_sigma).to_event(2))


    def get_embeddings(self):
        z_mu, _ = self.encoder.eval()(self.x, self.adj_norm)
        # Put encoder back into training mode
        self.encoder.train()
        return z_mu
