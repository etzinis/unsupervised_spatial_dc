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
                 n_train_mixtures=5000,
                 n_test_mixtures=2000,
                 n_val_mixtures=1000,
                 genders_mixtures=None):
        super(RandomCombinations,
              self).__init__(audio_dataset_name=audio_dataset_name)

        data_dic = self.data_loader.load()
        all_speakers = {'train': list(data_dic['train'].keys())}
        n_test_speakers = int(len(data_dic['test']) / 2)
        all_test_speakers = list(data_dic['test'].keys())
        all_speakers['test'] = all_test_speakers[n_test_speakers:]
        all_speakers['val'] = all_test_speakers[0:n_test_speakers]

        

        # print(len(train_data))
        # for speaker, speaker_info in train_data.items():
        #     print("{} {} {}".format(speaker,
        #                             speaker_info["gender"],
        #                             len(speaker_info['sentences'])))
        #     from pprint import pprint
        #     pprint(speaker_info)
        input()


if __name__ == "__main__":
    timit_random_combs = RandomCombinations(audio_dataset_name="timit")
    input()
    print(timit_random_combs)
    del timit_random_combs
    input()
