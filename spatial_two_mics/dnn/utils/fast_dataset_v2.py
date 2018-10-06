"""!
@brief A dataset creation which is compatible with pytorch framework
and much faster in loading time depending on the new version of
loading only the appropriate files that might be needed

@author Efthymios Tzinis {etzinis2@illinois.edu}
@copyright University of illinois at Urbana Champaign
"""

import torch
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
           '../../../')
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
                 force_delays=[-1, 1],
                 get_top=None,
                 labels_mask='duet',
                 **kwargs):

        self.dataset_params = {
            'dataset': dataset,
            'n_samples': n_samples,
            'n_sources': n_sources,
            'genders': genders,
            'force_delays': force_delays
        }

        if labels_mask == 'duet' or labels_mask == 'ground_truth':
            self.selected_mask = labels_mask
        else:
            raise NotImplementedError("There is no available mask "
                                      "called: {}".format(labels_mask))
        self.partition = partition

        dataset_name = dataset_storage.create_dataset_name(
            self.dataset_params)

        self.dataset_dirpath = os.path.join(
            config.DATASETS_DIR,
            dataset_name,
            partition)

        self.dataset_stats_path = self.dataset_dirpath + '_stats'

        if not os.path.isdir(self.dataset_dirpath):
            raise IOError("Dataset folder {} not found!".format(
                self.dataset_dirpath))
        else:
            print("Loading files from {} ...".format(
                self.dataset_dirpath))

        self.mixture_folders = glob2.glob(os.path.join(
            self.dataset_dirpath, '*'))
        if get_top is not None:
            self.mixture_folders = self.mixture_folders[:get_top]

        self.n_samples = len(self.mixture_folders)

        # preprocess -- store all absolute spectra values for faster
        # loading during run time
        self.store_directly_abs_spectra()

    def __len__(self):
        return self.n_samples

    def __getitem__(self, idx):
        """!
        Depending on the selected partition it returns accordingly
        the following objects:

        if self.partition == 'train':
            (abs_tfs, selected_mask)
        else if partition == 'test' or 'val'
            (abs_tfs, selected_mask, wavs_list, real_tfs, imag_tfs)"""
        mix_folder = self.mixture_folders[idx]
        try:
            abs_tfs = joblib.load(os.path.join(mix_folder, 'abs_tfs'))
        except:
            raise IOError("Failed to load data from path: {} "
                          "for absolute spectra.".format(mix_folder))


        try:
            if self.selected_mask == 'duet':
                mask = joblib.load(os.path.join(mix_folder,
                                                'soft_labeled_mask'))
            else:
                mask = joblib.load(os.path.join(mix_folder,
                                                'ground_truth_mask'))
        except:
            raise IOError("Failed to load data from path: {} "
                          "for tf label masks".format(mix_folder))

        if self.partition == 'train':
            return abs_tfs, mask

        try:
            real_p = os.path.join(mix_folder, 'real_tfs')
            imag_p = os.path.join(mix_folder, 'imag_tfs')
            wavs_p= os.path.join(mix_folder, 'wavs')
            real_tfs = joblib.load(real_p)
            imag_tfs = joblib.load(imag_p)
            wavs_list = joblib.load(wavs_p)
        except:
            raise IOError("Failed to load data from path: {} "
                          "for real, imag tf of the mixture and "
                          "wavs".format(mix_folder))

        return abs_tfs, mask, wavs_list, real_tfs, imag_tfs

    def store_directly_abs_spectra(self):
        for mix_folder in self.mixture_folders:
            try:
                real_p = os.path.join(mix_folder, 'real_tfs')
                imag_p = os.path.join(mix_folder, 'imag_tfs')
                real_tfs = joblib.load(real_p)
                imag_tfs = joblib.load(imag_p)
            except:
                raise IOError("Failed to load data from path: {} "
                              "using joblib.".format(mix_folder))
            abs_tfs = np.abs(real_tfs + 1j * imag_tfs)
            abs_p = os.path.join(mix_folder, 'abs_tfs')
            try:
                joblib.dump(abs_tfs, abs_p, compress=0)
            except:
                raise IOError("Failed to save absolute value of "
                              "spectra in path: {}".format(abs_p))

    def extract_stats(self):
        if not os.path.lexists(self.dataset_stats_path):
            mean = 0.
            std = 0.
            for mix_folder in self.mixture_folders:
                try:
                    abs_p = os.path.join(mix_folder, 'abs_tfs')
                    abs_tfs = joblib.load(abs_p)
                except:
                    raise IOError("Failed to load absolute tf "
                                  "representation from path: {} "
                                  "using joblib.".format(abs_p))

                mean += np.mean(np.mean(abs_tfs))
                std += np.std(abs_tfs)
            mean /= self.__len__()
            std /= self.__len__()

            #     store them for later usage
            joblib.dump((mean, std), self.dataset_stats_path)
            print("Saving dataset mean and variance in: {}".format(
                self.dataset_stats_path))
        else:
            mean, std = joblib.load(self.dataset_stats_path)

        return mean, std


def get_data_generator(args,
                       return_stats=False,
                       get_top=None):
    data = PytorchMixtureDataset(**args.__dict__,
                                 get_top=get_top)
    generator_params = {'batch_size': args.batch_size,
                        'shuffle': True,
                        'num_workers': args.num_workers,
                        'drop_last': True}
    data_generator = DataLoader(data,
                                **generator_params,
                                pin_memory=False)
    n_batches = int(len(data) / args.batch_size)
    if return_stats:
        mean, std = data.extract_stats()
        return data_generator, mean, std, n_batches
    else:
        return data_generator, n_batches


def concatenate_for_masks(masks, n_sources, batch_size):
    # create 3d masks for each source
    batch_list = []
    for b in torch.arange(batch_size):
        sources_list = []
        for i in torch.arange(n_sources):
            source_mask = masks[b, :, :] == int(i)
            sources_list.append(source_mask)

        sources_tensor = torch.stack(sources_list,
                                     dim=n_sources)
        batch_list.append(sources_tensor)
    return torch.stack(batch_list, dim=0)


def initialize_and_copy_masks(masks, n_sources, batch_size, device):
    new_masks = torch.empty((batch_size,
                             masks.shape[1],
                             masks.shape[2],
                             n_sources),
                            dtype=torch.uint8)
    new_masks.to(device)
    for i in torch.arange(n_sources):
        new_masks[:, :, :, i] = masks[:, :, :] == int(i)

    return new_masks


def example_of_usage(args):
    import time

    training_data = PytorchMixtureDataset(**args.__dict__)
    mean, std = training_data.extract_stats()
    generator_params = {'batch_size': 128,
                        'shuffle': True,
                        'num_workers': 1,
                        'drop_last': True}
    training_generator = DataLoader(training_data, **generator_params)
    device = torch.device("cuda")

    timing_dic = {}
    n_sources = 2

    batch_now = time.time()
    # just iterate over the data
    for batch_data in training_generator:
        timing_dic['Loading batch'] = time.time() - batch_now
        batch_now = time.time()

        before = time.time()
        (abs_tfs, masks) = batch_data
        now = time.time()
        timing_dic['Loading from disk'] = now-before

        before = time.time()
        input_tf, masks_tf = abs_tfs.to(device), masks.to(device)
        now = time.time()
        timing_dic['Loading to GPU'] = now - before


        before = time.time()
        duet_stack = concatenate_for_masks(masks,
                                           n_sources,
                                           generator_params['batch_size'])
        now = time.time()
        timing_dic['Stacking in appropriate dimensions the masks'] = \
            now - before

        before = time.time()
        duet_copy = initialize_and_copy_masks(masks,
                                              n_sources,
                                              generator_params[
                                                  'batch_size'],
                                              device)
        now = time.time()
        timing_dic['Initializing and copying for masks'] = now - before

        pprint(timing_dic)
        batch_now = time.time()


def get_args():
    """! Command line parser """
    parser = argparse.ArgumentParser(description='Pytorch Fast Dataset '
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


