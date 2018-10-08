"""!
@brief Loss functions for low rank approximations of an ideal
affinity mask

@author Efthymios Tzinis {etzinis2@illinois.edu}
@copyright University of illinois at Urbana Champaign
"""
import torch
import torch.nn as nn


def frobenius_naive(vs, ys):
    """! Computing naively the loss function between embedding
    vectors vs and ideal affinity matrices ys

    :param vs: size: batch_size x n_elements x embedded_features
    :param ys: One hot tensor corresponding to 1 where a specific
    class is the label for one element or 0 otherwise and its size:
    batch_size x n_elements x n_classes
    :return: The computed loss of these two tensors
    """
    loss = torch.mean(torch.norm(torch.norm(
           torch.matmul(vs, vs.permute(0, 2, 1)) -
           torch.matmul(ys, ys.permute(0, 2, 1)), 2, 1), 2, 1)**2)

    return loss


def efficient_frobenius(vs, ys, eps=10e-12):
    ys_T = ys.permute(0, 2, 1)
    vs_T = vs.permute(0, 2, 1)
    summed_y_T = ys_T.sum(dim=2).unsqueeze(-1)
    d = torch.bmm(ys, summed_y_T)
    d_m1_2 = torch.reciprocal(torch.sqrt(d) + eps)

    # print("Psola")
    # print((torch.bmm(vs_T, vs * d_m1_2)**2).shape)

    est_loss = (torch.bmm(vs_T, vs * d_m1_2) ** 2).sum()
    union_loss = (torch.bmm(vs_T, ys * d_m1_2) ** 2).sum()
    true_loss = (torch.bmm(ys_T, ys * d_m1_2) ** 2).sum()
    total_loss = est_loss - 2. * union_loss + true_loss
    # print(total_loss.shape)
    # print(est_loss.shape)
    # print(est_loss)

    print(union_loss)
    uni_loss = (torch.bmm(ys_T, vs * d_m1_2) ** 2).sum()
    print(uni_loss)

    return total_loss / vs.size(0)


def paris_naive(vs, ys):
    """! Computing naively the loss function between embedding
    vectors vs and ideal affinity matrices ys

    :param vs: size: batch_size x n_elements x embedded_features
    :param ys: One hot tensor corresponding to 1 where a specific
    class is the label for one element or 0 otherwise and its size:
    batch_size x n_elements x n_classes
    :return: The computed loss of these two tensors
    """
    loss = torch.mean(torch.matmul(vs.transpose(1, 2), vs) ** 2) \
         - 2. * torch.mean(torch.matmul(vs.transpose(1, 2), ys) ** 2) \
         + torch.mean(torch.matmul(ys.transpose(1, 2), ys) ** 2)
    return loss



def thymios_naive(vs, ys):
    """! Computing naively the loss function between embedding
    vectors vs and ideal affinity matrices ys

    :param vs: size: batch_size x n_elements x embedded_features
    :param ys: One hot tensor corresponding to 1 where a specific
    class is the label for one element or 0 otherwise and its size:
    batch_size x n_elements x n_classes
    :return: The computed loss of these two tensors
    """
    l = torch.sqrt((torch.matmul(vs.transpose(1, 2), vs) ** 2).sum()) \
        - 2.*torch.sqrt((torch.matmul(vs.transpose(1, 2), ys) **2).sum()) \
        + torch.sqrt((torch.matmul(ys.transpose(1, 2), ys) ** 2).sum())
    return l / vs.size(0)


def naive(vs, ys):
    """! Computing naively the loss function between embedding
    vectors vs and ideal affinity matrices ys

    :param vs: size: batch_size x n_elements x embedded_features
    :param ys: One hot tensor corresponding to 1 where a specific
    class is the label for one element or 0 otherwise and its size:
    batch_size x n_elements x n_classes
    :return: The computed loss of these two tensors
    """
    loss = (torch.matmul(vs.transpose(1, 2), vs) ** 2).sum() \
         - 2. * (torch.matmul(vs.transpose(1, 2), ys) ** 2).sum() \
         + (torch.matmul(ys.transpose(1, 2), ys) ** 2).sum()
    return loss / vs.size(0)
    # return loss


def diagonal(embedding, assignments):
    batch_size, sequence_length, num_frequencies, embedding_size = embedding.size()
    _, _, _, num_sources = assignments.size()
    embedding = embedding.view(-1, embedding.size()[-1])
    assignments = assignments.view(-1, assignments.size()[-1])

    class_weights = nn.functional.normalize(torch.sum(assignments.detach(), dim=-2), p=1, dim=-1).unsqueeze(0)
    class_weights = 1.0 / (torch.sqrt(class_weights) + 1e-7)
    weights = torch.matmul(assignments.detach(), class_weights.transpose(1, 0))
    # norm = torch.sum(weights**2)**2
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
    # loss = loss / norm
    return loss

