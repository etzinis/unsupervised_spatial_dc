"""!
@brief A dataset creation which is compatible with pytorch framework
and much faster in loading time depending on the new version of
loading only the appropriate files that might be needed. Moreover
this dataset has minimal input argument requirements in order to be
more user friendly.

@author Efthymios Tzinis {etzinis2@illinois.edu}
@copyright University of illinois at Urbana Champaign
"""

import os
import glob2
import numpy as np
from sklearn.externals import joblib
from torch.utils.data import Dataset, DataLoader


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
                 dataset_dir,
                 partition='train',
                 get_top=None,
                 labels_mask='duet',
                 **kwargs):
        """!
        Input dataset dir should have the following structure:
        ./dataset_dir
            ./train
            ./test
            ./val
        """

        self.dataset_dirpath = os.path.join(dataset_dir,
                                            partition)
        self.dataset_stats_path = self.dataset_dirpath + '_stats'
        self.partition = partition

        if labels_mask == 'duet' or labels_mask == 'ground_truth':
            self.selected_mask = labels_mask
        else:
            raise NotImplementedError("There is no available mask "
                                      "called: {}".format(labels_mask))

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
            wavs_list = np.array(wavs_list)
        except:
            raise IOError("Failed to load data from path: {} "
                          "for real, imag tf of the mixture and "
                          "wavs".format(mix_folder))

        return abs_tfs, mask, wavs_list, real_tfs, imag_tfs

    def store_directly_abs_spectra(self):
        for mix_folder in self.mixture_folders:
            abs_p = os.path.join(mix_folder, 'abs_tfs')
            if os.path.lexists(abs_p):
                continue

            try:
                real_p = os.path.join(mix_folder, 'real_tfs')
                imag_p = os.path.join(mix_folder, 'imag_tfs')
                real_tfs = joblib.load(real_p)
                imag_tfs = joblib.load(imag_p)
            except:
                raise IOError("Failed to load data from path: {} "
                              "using joblib.".format(mix_folder))
            abs_tfs = np.abs(real_tfs + 1j * imag_tfs)
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


def get_data_generator(dataset_dir,
                       partition='train',
                       num_workers=1,
                       return_stats=False,
                       get_top=None,
                       batch_size=1,
                       return_n_batches=True,
                       labels_mask='duet'):
    data = PytorchMixtureDataset(dataset_dir,
                                 partition=partition,
                                 get_top=get_top,
                                 labels_mask=labels_mask)
    generator_params = {'batch_size': batch_size,
                        'shuffle': True,
                        'num_workers': num_workers,
                        'drop_last': True}
    data_generator = DataLoader(data,
                                **generator_params,
                                pin_memory=False)

    results = [data_generator]

    if return_stats:
        mean, std = data.extract_stats()
        results += [mean, std]

    if return_n_batches:
        n_batches = int(len(data) / batch_size)
        results.append(n_batches)

    return results
