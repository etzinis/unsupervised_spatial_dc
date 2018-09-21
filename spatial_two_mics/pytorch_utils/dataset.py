"""!
@brief A dataset creation which is compatible with pytorch framework

@author Efthymios Tzinis {etzinis2@illinois.edu}
@copyright University of illinois at Urbana Champaign
"""

import argparse
import os
import sys
import glob2
import numpy as np
from sklearn.externals import joblib
import scipy.io.wavfile as wavfile
from torch.utils.data import Dataset, DataLoader
from pprint import pprint

root_dir = os.path.join(
           os.path.dirname(os.path.realpath(__file__)),
           '../../')
sys.path.insert(0, root_dir)
import spatial_two_mics.utils.audio_mixture_constructor as \
    mixture_creator
import spatial_two_mics.config as config
import spatial_two_mics.data_generator.dataset_storage as \
    dataset_storage


class PytorchMixtureDataset(Dataset):
    """
    This is a general compatible class for pytorch datasets.

    @note Each instance of the dataset should be stored using
    joblib.dump() and this is the way that it would be returned.
    After some transformations.

    The path of all datasets should be defined inside config.
    All datasets should be formatted with appropriate subfolders of
    train / test and val and under them there should be all the
    available files.
    """
    def __init__(self,
                 dataset='timit',
                 partition='train',
                 n_samples=[512, 128, 256],
                 n_sources=2,
                 genders=['f', 'm'],
                 n_fft=512,
                 win_len=512,
                 hop_length=128,
                 mixture_duration=2.0,
                 force_delays=[-1, 1]):

        self.dataset_params = {
            'dataset': dataset,
            'n_samples': n_samples,
            'n_sources': n_sources,
            'genders': genders,
            'force_delays': force_delays
        }
        dataset_name = dataset_storage.create_dataset_name(
                                       self.dataset_params)

        self.dataset_dirpath = os.path.join(
                               config.DATASETS_DIR,
                               dataset_name,
                               partition)

        if not os.path.isdir(self.dataset_dirpath):
            raise IOError("Dataset folder {} not found!".format(
                          self.dataset_dirpath))
        else:
            print("Loading files from {} ...".format(
                  self.dataset_dirpath))

        self.data_paths = glob2.glob(os.path.join(self.dataset_dirpath,
                                                  '*'))
        self.n_samples = len(self.data_paths)

        self.mix_creator = mixture_creator.AudioMixtureConstructor(
                           n_fft=n_fft,
                           win_len=win_len,
                           hop_len=hop_length,
                           mixture_duration=mixture_duration,
                           force_delays=force_delays)

    def __len__(self):
        return self.n_samples

    def __getitem__(self, idx):
        file_path = self.data_paths[idx]
        try:
            mixture_info = joblib.load(file_path)
        except:
            raise IOError("Failed to load data from path: {} "
                          "using joblib.".format(file_path))

        tf_info = self.mix_creator.construct_mixture(mixture_info)
        mixture_tf = tf_info['m1_tf']
        abs_tf = abs(mixture_tf)
        real_tf = np.real(mixture_tf)
        imag_tf = np.imag(mixture_tf)

        # assert (real_tf + 1j * imag_tf == mixture_tf).all()

        duet_mask = None
        ground_truth_mask = None
        try:
            duet_mask = mixture_info['soft_labeled_mask']
        except:
            raise KeyError("Mixture info does not have a soft label "
                           "attribute inferred by duet algorithm")

        try:
            ground_truth_mask = mixture_info['ground_truth_mask']
        except:
            raise KeyError("Mixture info does not have a ground truth "
                           "mask inferred by the most dominant source "
                           "in each TF bin.")

        sources_raw = np.array(tf_info['sources_raw'])
        amplitudes = np.array(mixture_info['positions']['amplitudes'])
        n_sources = len(sources_raw)

        return (abs_tf, real_tf, imag_tf,
                duet_mask, ground_truth_mask,
                sources_raw, amplitudes, n_sources)


def example_of_usage(args):
    import torch
    import time

    training_data = PytorchMixtureDataset(**args.__dict__)

    generator_params = {'batch_size': 64,
                        'shuffle': True,
                        'num_workers': 6,
                        'drop_last': True}
    training_generator = DataLoader(training_data, **generator_params)
    device = torch.device("cuda")

    timing_dic = {}

    # just iterate over the data
    for batch_data in training_generator:

        before = time.time()
        (abs_tfs, real_tfs, imag_tfs,
         duet_masks, ground_truth_masks,
         sources_raw, amplitudes, n_sources) = batch_data
        now = time.time()
        timing_dic['Loading from disk'] = now-before

        before = time.time()
        input_tf, mask_tf = abs_tfs.to(device), duet_masks.to(device)
        now = time.time()
        timing_dic['Loading to GPU'] = now - before

        before = time.time()
        # create 3d masks for each source
        batch_list = []
        for b in torch.arange(generator_params['batch_size']):
            sources_list = []
            for i in torch.arange(args.n_sources):
                source_mask = duet_masks == int(i)
                sources_list.append(source_mask)

            sources_tensor = torch.stack(sources_list,
                                         dim=args.n_sources)
            # print("Concatenated Source Vector is in shape: {}"
            #       "".format(sources_tensor.shape))
            batch_list.append(sources_tensor)
        result_tensor = torch.stack(batch_list, dim=0)
        # print("Concatenated Batch Tensor is in shape: {}"
        #       "".format(result_tensor.shape))

        now = time.time()
        timing_dic['Stacking in appropriate dimensions the masks'] = \
            now - before

        # torch.cuda.empty_cache()
        pprint(timing_dic)


def get_args():
    """! Command line parser """
    parser = argparse.ArgumentParser(description='Pytorch Dataset '
                                                 'Loader')
    parser.add_argument("--dataset", type=str,
                        help="Dataset name", default="timit")
    parser.add_argument("--n_sources", type=int,
                        help="How many sources in each mix", default=2)
    parser.add_argument("--n_samples", type=int, nargs='+',
                        help="How many samples do u want to be "
                             "created for train test val",
                        required=True)
    parser.add_argument("--genders", type=str, nargs='+',
                        help="Genders that will correspond to the "
                             "genders in the mixtures",
                        default=['m', 'f'])
    parser.add_argument("-f", "--force_delays", nargs='+', type=int,
                        help="""Whether you want to force integer 
                        delays of +- 1 in the sources e.g.""",
                        default=[-1,1])
    return parser.parse_args()


if __name__ == "__main__":
    args = get_args()
    example_of_usage(args)


