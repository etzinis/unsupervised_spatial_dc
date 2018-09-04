"""!
@brief A dataset creation which is compatible with pytorch framework

@author Efthymios Tzinis {etzinis2@illinois.edu}
@copyright University of illinois at Urbana Champaign
"""

import os
import sys
from torch.utils.data import Dataset, DataLoader

root_dir = os.path.join(
           os.path.dirname(os.path.realpath(__file__)),
           '../../')
sys.path.insert(0, root_dir)

import spatial_two_mics.data_loaders.timit as timit_loader


class PytorchCompatibleDataset(Dataset):
    """
    This is a general compatible class for pytorch datasets all other
    subclasses should inherit from this one and implement a sampler
    in order to create the appropriate combinations of mixtures
    """
    def __init__(self,
                 audio_dataset_name="timit"):
        if audio_dataset_name.lower() == "timit":
            self.data_loader = timit_loader.TimitLoader()
        else:
            raise NotImplementedError("Dataset Loader: {} is not yet "
                  "implemented.".format(audio_dataset_name))


class RandomCombinations(PytorchCompatibleDataset):
    def __init__(self,
                 audio_dataset_name="timit",
                 gender_mixtures=None,
                 n_mixtures=None,
                 n_mixed_sources=None,
                 genders_mixtures=None,
                 return_2_sets=False,
                 subset_of_speakers='train'):
        super(RandomCombinations,
              self).__init__(audio_dataset_name=audio_dataset_name)

        data_dic = self.data_loader.load()
        available_speakers = self.get_available_speakers(
                                  data_dic,
                                  subset_of_speakers,
                                  genders_mixtures)
        print(available_speakers)

    @staticmethod
    def get_available_speakers(data_dic,
                               subset_of_speakers,
                               genders_mixtures):
        try:
            available_speakers = sorted(list(data_dic[
                                 subset_of_speakers].keys()))
        except KeyError:
            print("Subset: {} not available".format(subset_of_speakers))
            raise KeyError

        valid_genders = [(g in ['f', 'm']) for g in genders_mixtures]
        assert valid_genders, ('Valid genders for mixtures are f and m')

        valid_speakers = []
        for speaker in available_speakers:
            if data_dic[subset_of_speakers][speaker]['gender'] in  \
                    genders_mixtures:
                valid_speakers.append(speaker)

        return valid_speakers

if __name__ == "__main__":
    timit_random_combs = RandomCombinations(
                         audio_dataset_name="timit",
                         genders_mixtures=['m'],
                         subset_of_speakers='test',
                         n_mixtures=3)

