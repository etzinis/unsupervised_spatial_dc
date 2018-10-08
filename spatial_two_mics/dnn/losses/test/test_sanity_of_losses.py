"""!
@brief Testing the sanity of the losses comparing to naive
implementations

@author Efthymios Tzinis {etzinis2@illinois.edu}
@copyright University of illinois at Urbana Champaign
"""
import sys
import numpy as np
import torch
from pprint import pprint
sys.path.append('../')
import affinity_approximation as losses


def numpy_naive(vs, ys):
    frobenius_np = np.mean(
        np.array([np.linalg.norm(vs[b].dot(vs[b].T) -
                                 ys[b].dot(ys[b].T))**2
                  for b in np.arange(vs.shape[0])]))
    return frobenius_np


if __name__ == "__main__":
    batch_size = 1
    num_tfs = 100
    embedding_depth = 10
    n_sources = 2
    vs_np = np.random.rand(batch_size, num_tfs, embedding_depth)
    ys_np = np.abs(np.random.rand(batch_size, num_tfs, n_sources))
    vs = torch.from_numpy(vs_np)
    ys = torch.from_numpy(ys_np)

    np_frobenius = numpy_naive(vs_np, ys_np)
    naive_torch_frobenius = losses.frobenius_naive(vs, ys).data.numpy()
    #
    print("Numpy Frobenius: {}".format(np_frobenius))
    print("Naive Torch Frobenius: {}".format(naive_torch_frobenius))

    assert np.abs(np_frobenius -
                  naive_torch_frobenius) < 10e-5, 'Naive ' \
                    'implementations of Frobenius norm should be equal'



    efficient_frobenius = losses.efficient_frobenius(vs, ys)
    print("Efficient Frobenius: {}".format(efficient_frobenius))

    # assert np.abs(np_frobenius -
    #               efficient_frobenius) < 10e-5, 'Efficient == Naive '

    paris_wtf = losses.naive(vs, ys)
    print("Paris wtf: {}".format(paris_wtf))
