"""!
@brief Loss functions for low rank approximations of an ideal
affinity mask

@author Efthymios Tzinis {etzinis2@illinois.edu}
@copyright University of illinois at Urbana Champaign
"""
import torch

def naive(vs, ys):
    """! Computing naively the loss function between embedding
    vectors vs and ideal affinity matrices ys

    :param vs: size: batch_size x n_elements x embedded_features
    :param ys: One hot tensor corresponding to 1 where a specific
    class is the label for one element or 0 otherwise and its size:
    batch_size x n_elements x n_classes
    :return: The computed loss of these two tensors
    """
    loss = torch.sqrt(torch.mean(torch.matmul(
           vs.transpose(1, 2), vs) ** 2)) \
         - 2. * torch.sqrt(torch.mean(torch.matmul(
           vs.transpose(1, 2), ys) ** 2)) \
         + torch.sqrt(torch.mean(torch.matmul(
           ys.transpose(1, 2), ys) ** 2))
    return loss / vs.size(0)
