"""!
@brief Loss functions for low rank approximations of an ideal
affinity mask

@author Efthymios Tzinis {etzinis2@illinois.edu}
@copyright University of illinois at Urbana Champaign
"""
import torch
import torch.nn as nn


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


def diagonal(embedding, assignments):
    batch_size, sequence_length, num_frequencies, embedding_size = embedding.size()
    _, _, _, num_sources = assignments.size()
    embedding = embedding.view(-1, embedding.size()[-1])
    assignments = assignments.view(-1, assignments.size()[-1])

    class_weights = nn.functional.normalize(torch.sum(assignments.detach(), dim=-2), p=1, dim=-1).unsqueeze(0)
    class_weights = 1.0 / (torch.sqrt(class_weights) + 1e-7)
    weights = torch.matmul(assignments.detach(), class_weights.transpose(1, 0))
    norm = torch.sum(weights**2)**2
    assignments = assignments * weights.repeat(1, assignments.size()[-1])
    embedding = embedding * weights.repeat(1, embedding.size()[-1])

    embedding = embedding.view(batch_size, sequence_length*num_frequencies, embedding_size)
    assignments = assignments.view(batch_size, sequence_length*num_frequencies, num_sources)

    embedding_transpose = embedding.permute(0, 2, 1)
    assignments_transpose = assignments.permute(0, 2, 1)

    loss_est = torch.sum(torch.matmul(embedding_transpose, embedding)**2)
    loss_est_true = torch.sum(torch.matmul(embedding_transpose, assignments)**2)
    loss_true = torch.sum(torch.matmul(assignments_transpose, assignments)**2)
    loss = loss_est - 2*loss_est_true + loss_true
    loss = loss / norm
    return loss

