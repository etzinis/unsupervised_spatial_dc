"""!
@brief Pytorch data tensors manipulations functions

@author Efthymios Tzinis {etzinis2@illinois.edu}
@copyright University of illinois at Urbana Champaign
"""

import torch


def one_hot_3Dmasks(index_ys, n_classes):
    """! Converting a matrix of float labels for each class to a one
    hot vector of the same dimension plus the extra of one-hot
    correspondence

    :param index_ys: mask 3d tensor with integer labels
    :param n_classes: integer
    :return: whatever diomensions x n_classes => 1 hot correspondence
    """
    clustered_ys = index_ys.unsqueeze(-1).long()

    one_hot = torch.cuda.FloatTensor(clustered_ys.size(0),
                                     clustered_ys.size(1),
                                     clustered_ys.size(2),
                                     n_classes).zero_()

    return one_hot.scatter_(3, clustered_ys, 1).cuda()

