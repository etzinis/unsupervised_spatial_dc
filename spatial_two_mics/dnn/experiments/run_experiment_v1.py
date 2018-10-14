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
import spatial_two_mics.dnn.utils.fast_dataset_v2 as fast_data_gen
import spatial_two_mics.dnn.utils.data_conversions as converters
import spatial_two_mics.dnn.utils.experiment_command_line_parser_v2 as \
    parser
import spatial_two_mics.dnn.utils.update_history as update_history
from progress.bar import ChargingBar
import spatial_two_mics.dnn.evaluation.naive_evaluation_numpy as \
    numpy_eval
from sklearn.cluster import KMeans


def train(args,
          model,
          training_generator,
          optimizer,
          mean_tr,
          std_tr,
          epoch,
          history,
          n_batches):
    model.train()
    timing_dic = {'Loading batch': 0.,
                  'Transformations and Forward': 0.,
                  'Loss Computation and Backprop': 0.}
    before = time.time()
    bar = ChargingBar("Training for epoch: {}...".format(epoch),
                      max=n_batches)
    for batch_data in training_generator:
        (abs_tfs, masks) = batch_data
        timing_dic['Loading batch'] += time.time() - before
        before = time.time()
        input_tfs, index_ys = abs_tfs.cuda(), masks.cuda()
        # the input sequence is determined by time and not freqs
        # before: input_tfs = batch_size x (n_fft/2+1) x n_timesteps
        input_tfs = input_tfs.permute(0, 2, 1).contiguous()
        index_ys = index_ys.permute(0, 2, 1).contiguous()

        # normalize with mean and variance from the training dataset
        input_tfs -= mean_tr
        input_tfs /= std_tr

        # index_ys = index_ys.permute(0, 2, 1).contiguous()
        one_hot_ys = converters.one_hot_3Dmasks(index_ys,
                                                args.n_sources)

        optimizer.zero_grad()
        vs = model(input_tfs)

        flatened_ys = one_hot_ys.view(one_hot_ys.size(0),
                                      -1,
                                      one_hot_ys.size(-1)).cuda()

        timing_dic['Transformations and Forward'] += time.time() - \
                                                     before
        before = time.time()
        loss = affinity_losses.paris_naive(vs, flatened_ys)
        # loss = affinity_losses.diagonal(vs.view(vs.size(0),
        #                                         one_hot_ys.size(1),
        #                                         one_hot_ys.size(2),
        #                                         vs.size(-1)),
        #                                 one_hot_ys.cuda())

        loss.backward()
        nn.utils.clip_grad_norm(model.parameters(), 100.)
        optimizer.step()
        timing_dic['Loss Computation and Backprop'] += time.time() - \
                                                       before

        update_history.values_update([('loss', loss)],
                                     history, update_mode='batch')
        before = time.time()
        bar.next()
    bar.finish()

    pprint(timing_dic)


def eval(args,
         model,
         val_generator,
         mean_tr,
         std_tr,
         epoch,
         history,
         n_batches,
         k_means_obj):
    timing_dic = {'Standard Scaler': 0.,
                  'Kmeans': 0.,
                  'Dummy BSS evaluation': 0.}

    # make some evaluation
    model.eval()
    before = time.time()
    with torch.no_grad():
        bar = ChargingBar("Evaluating for epoch: {}...".format(epoch),
                          max=n_batches)
        before = time.time()
        for batch_data in val_generator:
            abs_tfs, masks, wavs_lists, real_tfs, imag_tfs = batch_data
            input_tfs = abs_tfs.cuda()
            # the input sequence is determined by time and not freqs
            # before: input_tfs = batch_size x (n_fft/2+1) x n_timesteps
            input_tfs = input_tfs.permute(0, 2, 1).contiguous()

            # normalize with mean and variance from the training dataset
            input_tfs -= mean_tr
            input_tfs /= std_tr

            vs = model(input_tfs)
            for b in np.arange(vs.size(0)):

                # possibly go into GPU ?
                # before = time.time()
                # embedding_features = z_scaler.fit_transform(
                #     vs[b, :, :].data.cpu().numpy())
                # timing_dic['Standard Scaler'] += time.time() - before

                embedding_features = vs[b, :, :].data.cpu().numpy()
                # embedding_features = masks[b, :, :].view(-1, 1).data.numpy()
                # embedding_labels = masks[b].data.numpy()
                # embedding_features = flatened_ys[b, :, :].data.cpu().numpy()



                # possibly perform kmeans on GPU?
                before = time.time()
                embedding_labels = np.array(k_means_obj.fit_predict(
                                            embedding_features))
                timing_dic['Kmeans'] += time.time() - before

                # possibly do it on GPU?
                before = time.time()
                sdr, sir, sar = numpy_eval.naive_cpu_bss_eval(
                    embedding_labels,
                    real_tfs[b].data.numpy(),
                    imag_tfs[b].data.numpy(),
                    wavs_lists[b].data.numpy(),
                    args.n_sources,
                    batch_index=b)
                timing_dic['Dummy BSS evaluation'] += time.time() - before

                update_history.values_update([('sdr', sdr),
                                              ('sir', sir),
                                              ('sar', sar)],
                                             history,
                                             update_mode='batch')

            bar.next()
        pprint(timing_dic)
        bar.finish()


def run_LSTM_experiment(args):
    visible_cuda_ids = ','.join(map(str, args.cuda_available_devices))
    os.environ["CUDA_VISIBLE_DEVICES"] = visible_cuda_ids

    (training_generator, mean_tr, std_tr, n_tr_batches) = \
        fast_data_gen.get_data_generator(args,
                                         return_stats=True)

    val_args = copy.copy(args)
    val_args.partition = 'val'
    val_generator, n_val_batches = \
        fast_data_gen.get_data_generator(val_args,
                                         get_top=args.n_eval)

    model = LSTM_enc.BLSTMEncoder(num_layers=args.n_layers,
                                  hidden_size=args.hidden_size,
                                  embedding_depth=args.embedding_depth,
                                  bidirectional=args.bidirectional)
    model = nn.DataParallel(model).cuda()

    optimizer = torch.optim.Adam(model.parameters(),
                                 lr=args.learning_rate,
                                 betas=(0.9, 0.999))

    k_means_obj = KMeans(n_clusters=2)
    # just iterate over the data
    history = {}
    for epoch in np.arange(args.epochs):

        train(args, model, training_generator, optimizer, mean_tr,
              std_tr, epoch, history, n_tr_batches)

        update_history.values_update([('loss', None)],
                                     history,
                                     update_mode='epoch')


        if epoch % args.evaluate_per == 0:
            eval(args, model, val_generator, mean_tr,
                 std_tr, epoch, history, n_val_batches, k_means_obj)

            update_history.values_update([('sdr', None),
                                          ('sir', None),
                                          ('sar', None)],
                                         history,
                                         update_mode='epoch')

        pprint(history['loss'][-1])
        pprint(history['sdr'][-1])
        pprint(history['sir'][-1])
        pprint(history['sar'][-1])
        print(
            "BEST SDR: {}, SIR: {}, SAR {}".format(max(history['sdr']),
                                                   max(history['sir']),
                                                   max(history['sar'])))


if __name__ == "__main__":
    args = parser.get_args()
    # run_LSTM_experiment(args)