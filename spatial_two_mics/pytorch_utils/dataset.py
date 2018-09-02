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


class PytorchDataset(Dataset):
    def __init__(self,
                 audio_dataset_name="timit"):
        if audio_dataset_name.lower() == "timit":
            self.data_loader = timit_loader.TimitLoader()
        else:
            raise NotImplementedError("Dataset Loader: {} is not yet "
                  "implemented.".format(audio_dataset_name))

        data_dic = self.data_loader.load()
        train_data = data_dic['train']
        print(len(train_data))
        for speaker, speaker_info in train_data.items():
            print("{} {} {}".format(speaker,
                                    speaker_info["gender"],
                                    len(speaker_info['sentences'])))
            from pprint import pprint
            pprint(speaker_info)


if __name__ == "__main__":
    timit_pytorch_comp = PytorchDataset(audio_dataset_name="timit")
    print(timit_pytorch_comp)