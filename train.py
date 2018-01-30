from __future__ import division
from __future__ import print_function

import numpy as np
import scipy.sparse as sp

import torch
from torch.autograd import Variable

import pyro
from pyro.infer import SVI
from pyro.optim import Adam

from utils import load_data, dotdict, eval_gae, make_sparse
from models import GAE
from preprocessing import mask_test_edges, preprocess_graph


def main(args):
    """ Train GAE """ 
    
    # Load data
    np.random.seed(1)
    adj, features = load_data(args.dataset_str)
    N, D = features.shape

    # Store original adjacency matrix (without diagonal entries)
    adj_orig = adj
    adj_train, train_edges, val_edges, val_edges_false, test_edges, test_edges_false = mask_test_edges(adj)

    # Some preprocessing
    adj_train_norm   = preprocess_graph(adj_train)
    adj_train_norm   = Variable(make_sparse(adj_train_norm))
    adj_train_labels = Variable(torch.FloatTensor(adj_train + sp.eye(adj_train.shape[0]).todense()))
    features         = Variable(torch.FloatTensor(np.array(features.todense())))

    data = {
        'adj_norm'  : adj_train_norm,
        'adj_labels': adj_train_labels,
        'features'  : features,
    }

    gae = GAE(data,
              n_hidden=32,
              n_latent=16,
              dropout=args.dropout)

    optimizer = Adam({"lr": args.lr, "betas": (0.95, 0.999)})

    svi = SVI(gae.model, gae.guide, optimizer, loss="ELBO")
    train_elbo    = []
    
    # test_elbo     = []
    # cost_val      = []
    # acc_val       = []

    val_roc_score = []

    # Full batch training loop
    for epoch in range(args.num_epochs):
        # initialize loss accumulator
        epoch_loss = 0.
        # do ELBO gradient and accumulate loss
        epoch_loss += svi.step()
        # report training diagnostics
        normalized_loss = epoch_loss / (2 * N * N)
        train_elbo.append(normalized_loss)

        # Training loss
        emb = gae.get_embeddings()
        accuracy, roc_curr, ap_curr = eval_gae(val_edges, val_edges_false, emb, adj_orig)
        val_roc_score.append(roc_curr)

        print("Epoch:", '%04d' % (epoch + 1),
              "train_loss=", "{:.5f}".format(normalized_loss),
              "train_acc=", "{:.5f}".format(accuracy), "val_roc=", "{:.5f}".format(roc_curr), "val_ap=", "{:.5f}".format(ap_curr))

    print("Optimization Finished!")

    # Test loss
    emb = gae.get_embeddings()
    accuracy, roc_score, ap_score = eval_gae(test_edges, test_edges_false, emb, adj_orig)
    print('Test Accuracy: ' + str(accuracy))
    print('Test ROC score: ' + str(roc_score))
    print('Test AP score: ' + str(ap_score))


if __name__ == '__main__':

    args = dotdict()
    args.seed = 2
    args.dropout = 0.5
    args.num_epochs = 400
    args.dataset_str = 'cora'
    args.lr = 0.01

    pyro.clear_param_store()
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    main(args)
