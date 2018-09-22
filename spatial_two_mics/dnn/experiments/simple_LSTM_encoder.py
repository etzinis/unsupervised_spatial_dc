"""!
@brief A simple experiment on how models, losses, etc should be used

@author Efthymios Tzinis {etzinis2@illinois.edu}
@copyright University of illinois at Urbana Champaign
"""

import argparse
import os
import sys
import torch
import time
import copy
from pprint import pprint
from torch.utils.data import DataLoader

root_dir = os.path.join(
    os.path.dirname(os.path.realpath(__file__)),
    '../../../')
sys.path.insert(0, root_dir)

import spatial_two_mics.dnn.models.simple_LSTM_encoder as LSTM_enc
import spatial_two_mics.dnn.losses as losses
import spatial_two_mics.pytorch_utils.dataset as data_generator


def check_device_model_laoding(model):
    device = 0
    print(torch.cuda.get_device_capability(device=device))
    print(torch.cuda.memory_allocated(device=device))
    print(torch.cuda.memory_cached(device=device))

    model = model.cuda()
    print(torch.cuda.get_device_properties(device=device).total_memory)
    print(torch.cuda.memory_allocated(device))
    print(torch.cuda.memory_cached(device))

    temp_model = copy.deepcopy(model)
    temp_model = temp_model.cuda()
    print(torch.cuda.max_memory_cached(device=device))
    print(torch.cuda.memory_allocated(device))
    print(torch.cuda.memory_cached(device))


def example_of_usage(args):

    cuda_id = "cuda:"+str(args.cuda_device)
    cuda_id = "cuda:0,1"

    training_generator = data_generator.get_data_generator(args)
    device = torch.device(cuda_id)
    timing_dic = {}

    before = time.time()
    model = LSTM_enc.BLSTMEncoder(num_layers=args.n_layers,
                                  hidden_size=args.hidden_size,
                                  embedding_depth=args.embedding_depth,
                                  bidirectional=args.bidirectional)
    timing_dic['Iitializing model'] = time.time() - before
    os.environ["CUDA_VISIBLE_DEVICES"] = "1,2,3"
    model = model.cuda()
    timing_dic['Transfering model to device'] = time.time() - before

    optimizer = torch.optim.Adam(model.parameters(),
                                 lr=args.learning_rate,
                                 betas=(0.9, 0.999))

    batch_now = time.time()
    # just iterate over the data
    for batch_data in training_generator:
        timing_dic['Loading batch'] = time.time() - batch_now
        batch_now = time.time()

        before = time.time()
        (abs_tfs, real_tfs, imag_tfs,
         duet_masks, ground_truth_masks,
         sources_raw, amplitudes, n_sources) = batch_data

        input_tfs, masks_tfs = abs_tfs.to(device), duet_masks.cuda()

        # the input sequence is determined by time and not freqs
        input_tfs = input_tfs.permute(0, 2, 1)

        optimizer.zero_grad()
        vs = model(input_tfs)


def get_args():
    """! Command line parser """
    parser = argparse.ArgumentParser(description='Pytorch Dataset '
                                                 'Loader')
    parser.add_argument("--dataset", type=str,
                        help="Dataset name",
                        default="timit")
    parser.add_argument("--n_sources", type=int,
                        help="How many sources in each mix",
                        default=2)
    parser.add_argument("--n_samples", type=int, nargs='+',
                        help="How many samples do u want to be "
                             "created for train test val",
                        default=[256, 64, 128])
    parser.add_argument("--genders", type=str, nargs='+',
                        help="Genders that will correspond to the "
                             "genders in the mixtures",
                        default=['m'])
    parser.add_argument("-f", "--force_delays", nargs='+', type=int,
                        help="""Whether you want to force integer 
                        delays of +- 1 in the sources e.g.""",
                        default=[-1, 1])
    parser.add_argument("-nl", "--n_layers", type=int,
                        help="""The number of layers of the LSTM 
                        encoder""", default=2)
    parser.add_argument("-ed", "--embedding_depth", type=int,
                        help="""The depth of the embedding""",
                        default=10)
    parser.add_argument("-hs", "--hidden_size", type=int,
                        help="""The size of the LSTM cells """,
                        default=10)
    parser.add_argument("-bs", "--batch_size", type=int,
                        help="""The number of samples in each batch""",
                        default=64)
    parser.add_argument("-cd", "--cuda_device", type=int,
                        help="""The Cuda device ID""",
                        default=0)
    parser.add_argument("--num_workers", type=int,
                        help="""The number of cpu workers for 
                        loading the data, etc.""", default=3)
    parser.add_argument("-lr", "--learning_rate", type=float,
                        help="""Initial Learning rate""", default=1e-3)
    parser.add_argument("--bidirectional", action='store_true',
                        help="""Bidirectional or not""")

    return parser.parse_args()


if __name__ == "__main__":
    args = get_args()
    example_of_usage(args)