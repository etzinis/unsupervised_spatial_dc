"""!
@brief Using the fast version of the dataset generator provide a
naive experimental setup for performing the experiment using also the
new command line argument parser.

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
import spatial_two_mics.dnn.utils.fast_dataset_v3 as fast_data_gen
import spatial_two_mics.dnn.utils.data_conversions as converters
import spatial_two_mics.dnn.utils.experiment_command_line_parser_v2 as \
    parser
import spatial_two_mics.dnn.utils.update_history as update_history
import spatial_two_mics.dnn.utils.model_logger as model_logger
from progress.bar import ChargingBar
import spatial_two_mics.dnn.evaluation.naive_evaluation_numpy as \
    numpy_eval
from sklearn.cluster import KMeans


def train(model,
          training_generator,
          optimizer,
          mean_tr,
          std_tr,
          epoch,
          history,
          n_batches,
          n_sources,
          training_labels=''):
    model.train()
    bar = ChargingBar("Training for epoch: {}...".format(epoch),
                      max=n_batches)
    for batch_data in training_generator:
        (abs_tfs, masks) = batch_data
        input_tfs, index_ys = abs_tfs.cuda(), masks.cuda()
        # the input sequence is determined by time and not freqs
        # before: input_tfs = batch_size x (n_fft/2+1) x n_timesteps
        input_tfs = input_tfs.permute(0, 2, 1).contiguous()
        index_ys = index_ys.permute(0, 2, 1).contiguous()

        # normalize with mean and variance from the training dataset
        input_tfs -= mean_tr
        input_tfs /= std_tr

        if training_labels == 'raw_phase_diff':
            flatened_ys = index_ys.view(index_ys.size(0), -1, 1)
        else:
            # index_ys = index_ys.permute(0, 2, 1).contiguous()
            one_hot_ys = converters.one_hot_3Dmasks(index_ys,
                                                    n_sources)
            flatened_ys = one_hot_ys.view(one_hot_ys.size(0),
                                          -1,
                                          one_hot_ys.size(-1)).cuda()

        optimizer.zero_grad()
        vs = model(input_tfs)


        loss = affinity_losses.paris_naive(vs, flatened_ys)

        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 100.)
        optimizer.step()

        update_history.values_update([('loss', loss)],
                                     history, update_mode='batch')
        bar.next()
    bar.finish()


def eval(model,
         val_generator,
         mean_tr,
         std_tr,
         epoch,
         history,
         n_batches,
         k_means_obj,
         n_sources,
         batch_size):

    model.eval()
    with torch.no_grad():
        bar = ChargingBar("Evaluating for epoch: {}...".format(epoch),
                          max=n_batches*batch_size)
        for batch_data in val_generator:
            abs_tfs, wavs_lists, real_tfs, imag_tfs = batch_data
            input_tfs = abs_tfs.cuda()
            # the input sequence is determined by time and not freqs
            # before: input_tfs = batch_size x (n_fft/2+1) x n_timesteps
            input_tfs = input_tfs.permute(0, 2, 1).contiguous()

            # normalize with mean and variance from the training dataset
            input_tfs -= mean_tr
            input_tfs /= std_tr

            vs = model(input_tfs)
            for b in np.arange(vs.size(0)):

                embedding_features = vs[b, :, :].data.cpu().numpy()

                embedding_labels = np.array(k_means_obj.fit_predict(
                                            embedding_features))

                sdr, sir, sar = numpy_eval.naive_cpu_bss_eval(
                    embedding_labels,
                    real_tfs[b].data.numpy(),
                    imag_tfs[b].data.numpy(),
                    wavs_lists[b].data.numpy(),
                    n_sources,
                    batch_index=b)

                update_history.values_update([('sdr', sdr),
                                              ('sir', sir),
                                              ('sar', sar)],
                                             history,
                                             update_mode='batch')

                bar.next()
        bar.finish()


def run_LSTM_experiment(args):
    visible_cuda_ids = ','.join(map(str, args.cuda_available_devices))
    os.environ["CUDA_VISIBLE_DEVICES"] = visible_cuda_ids

    (training_generator, mean_tr, std_tr, n_tr_batches, n_tr_sources) =\
    fast_data_gen.get_data_generator(args.train,
                                     partition='train',
                                     num_workers=args.num_workers,
                                     return_stats=True,
                                     get_top=args.n_train,
                                     batch_size=args.batch_size,
                                     return_n_batches=True,
                                     labels_mask=args.training_labels,
                                     return_n_sources=True)

    val_generator, n_val_batches, n_val_sources = \
    fast_data_gen.get_data_generator(args.val,
                                     partition='val',
                                     num_workers=args.num_workers,
                                     return_stats=False,
                                     get_top=args.n_val,
                                     batch_size=args.batch_size,
                                     return_n_batches=True,
                                     labels_mask=None,
                                     return_n_sources=True)

    model = LSTM_enc.BLSTMEncoder(num_layers=args.n_layers,
                                  hidden_size=args.hidden_size,
                                  embedding_depth=args.embedding_depth,
                                  bidirectional=args.bidirectional,
                                  dropout=args.dropout)
    model = nn.DataParallel(model).cuda()

    optimizer = torch.optim.Adam(model.parameters(),
                                 lr=args.learning_rate,
                                 betas=(0.9, 0.999))

    assert n_val_sources == n_tr_sources, "Number of sources in both " \
                                          "training and evaluation " \
                                          "should be equal while " \
                                          "training"
    k_means_obj = KMeans(n_clusters=n_tr_sources)
    # just iterate over the data
    history = {}
    for epoch in np.arange(args.epochs):

        train(model, training_generator, optimizer, mean_tr,
              std_tr, epoch, history, n_tr_batches, n_tr_sources,
              training_labels=args.training_labels)

        update_history.values_update([('loss', None)],
                                     history,
                                     update_mode='epoch')


        if epoch % args.eval_per == 0:
            eval(model, val_generator, mean_tr, std_tr, epoch,
                 history, n_val_batches, k_means_obj, n_val_sources,
                 args.batch_size)

            update_history.values_update([('sdr', None),
                                          ('sir', None),
                                          ('sar', None)],
                                         history,
                                         update_mode='epoch')

            # keep track of best performances so far
            epoch_performance_dic = {
                'sdr': history['sdr'][-1],
                'sir': history['sir'][-1],
                'sar': history['sar'][-1]
            }
            update_history.update_best_performance(
                           epoch_performance_dic, epoch, history,
                           buffer_size=args.save_best)


            # save the model if it is one of the best according to SDR
            if (history['sdr'][-1] >=
                history['best_performances'][-1][0]['sdr']):
                dataset_id = os.path.basename(args.train)

                model_logger.save(model,
                                  optimizer,
                                  args,
                                  epoch,
                                  epoch_performance_dic,
                                  dataset_id,
                                  mean_tr,
                                  std_tr,
                                  training_labels=args.training_labels)


        pprint(history['loss'][-1])
        pprint(history['best_performances'])


if __name__ == "__main__":
    args = parser.get_args()
    run_LSTM_experiment(args)