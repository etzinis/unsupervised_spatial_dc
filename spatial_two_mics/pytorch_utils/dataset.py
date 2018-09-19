"""!
@brief A dataset creation which is compatible with pytorch framework

@author Efthymios Tzinis {etzinis2@illinois.edu}
@copyright University of illinois at Urbana Champaign
"""

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


class PytorchMixtureDataset(Dataset):
    """
    This is a general compatible class for pytorch datasets.

    @note Each instance of the dataset should be stored using
    joblib.dump() and this is the way that it would be returned.
    After some transformations.
    """
    def __init__(self,
                 root_dir_path,
                 n_fft=512,
                 win_len=512,
                 hop_length=128,
                 mixture_duration=2.0,
                 force_delays=[-1, 1]):

        if not os.path.isdir(root_dir_path):
            raise IOError("Dataset folder {} not found!".format(
                          root_dir_path))
        self.root_path = root_dir_path
        self.data_paths = glob2.glob(os.path.join(self.root_path, '*'))
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
        mixture_tf = abs(tf_info['m1_tf'])

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

        return mixture_tf, duet_mask, ground_truth_mask


def example_of_usage(dataset_path):
    training_data = PytorchMixtureDataset(dataset_path)

    generator_params = {'batch_size': 10,
                        'shuffle': True,
                        'num_workers': 2}
    training_generator = DataLoader(training_data, **generator_params)

    # just iterate over the data
    for batch_data in training_generator:
        mixtures_tf, duet_masks, gt_masks = batch_data
        print(type(mixtures_tf))
        print(mixtures_tf.shape)
        print(duet_masks.shape)
        print(gt_masks.shape)
        break



if __name__ == "__main__":
    root_dir = '/home/thymios/data'
    data_parametered_name = 'timit_27000_9000_18000_2_m_-1taus1'
    data_parametered_name = 'timit_256_64_128_2_m_-1taus1'
    partition = 'train'
    dataset_path = os.path.join(root_dir,
                                data_parametered_name,
                                partition)
    example_of_usage(dataset_path)


