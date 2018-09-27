"""!
@brief A simple experiment on how LSTM converge

@author Efthymios Tzinis {etzinis2@illinois.edu}
@copyright University of illinois at Urbana Champaign
"""

import os
import sys
import torch
import time
import numpy as np
import copy
from pprint import pprint
import torch.nn as nn

root_dir = os.path.join(
    os.path.dirname(os.path.realpath(__file__)),
    '../../../')
sys.path.insert(0, root_dir)

import spatial_two_mics.dnn.models.simple_LSTM_encoder as LSTM_enc
import spatial_two_mics.dnn.losses.affinity_approximation as \
    affinity_losses
import spatial_two_mics.dnn.utils.dataset as data_generator
import spatial_two_mics.dnn.utils.data_conversions as converters
import spatial_two_mics.dnn.utils.experiment_command_line_parser as \
    parser
import spatial_two_mics.dnn.utils.update_history as update_history


def convergence_of_LSTM(args):
    visible_cuda_ids = ','.join(map(str, args.cuda_available_devices))
    os.environ["CUDA_VISIBLE_DEVICES"] = visible_cuda_ids

    training_generator = data_generator.get_data_generator(args)

    before = time.time()
    model = LSTM_enc.BLSTMEncoder(num_layers=args.n_layers,
                                  hidden_size=args.hidden_size,
                                  embedding_depth=args.embedding_depth,
                                  bidirectional=args.bidirectional)
    model = model.cuda()

    optimizer = torch.optim.Adam(model.parameters(),
                                 lr=args.learning_rate,
                                 betas=(0.9, 0.999))

    batch_size = args.batch_size
    # just iterate over the data
    history = {}
    for epoch in np.arange(args.epochs):
        print("Training for epoch: {}...".format(epoch))
        for batch_data in training_generator:
            (abs_tfs, real_tfs, imag_tfs,
             duet_masks, ground_truth_masks,
             sources_raw, amplitudes, n_sources) = batch_data

            input_tfs, index_ys = abs_tfs.cuda(), duet_masks.cuda()
            # the input sequence is determined by time and not freqs
            # before: input_tfs = batch_size x (n_fft/2+1) x n_timesteps
            input_tfs = input_tfs.permute(0, 2, 1).contiguous()
            index_ys = index_ys.permute(0, 2, 1).contiguous()

            one_hot_ys = converters.one_hot_3Dmasks(index_ys,
                                                    n_sources[0])

            optimizer.zero_grad()
            vs = model(input_tfs)

            flatened_ys = one_hot_ys.view(one_hot_ys.size(0),
                                          -1,
                                          one_hot_ys.size(-1)).cuda()
            naive_loss = affinity_losses.naive(vs, flatened_ys)
            naive_loss.backward()
            optimizer.step()
            print("Naive Loss: {}".format(naive_loss))

            update_history.values_update([('loss', naive_loss)],
                                         history,
                                         update_mode='batch')

        update_history.values_update([('loss', None)],
                                     history,
                                     update_mode='epoch')

        pprint(history)

if __name__ == "__main__":
    args = parser.get_args()
    convergence_of_LSTM(args)
