"""!
@brief A dataset creation which is compatible with pytorch framework

@author Efthymios Tzinis {etzinis2@illinois.edu}
@copyright University of illinois at Urbana Champaign
"""

import os
import sys
import glob2
from torch.utils.data import Dataset, DataLoader

root_dir = os.path.join(
           os.path.dirname(os.path.realpath(__file__)),
           '../../')
sys.path.insert(0, root_dir)


class PytorchMixtureDataset(Dataset):
    """
    This is a general compatible class for pytorch datasets.

    @note Each instance of the dataset should be stored using
    joblib.dump() and this is the way that it would be returned.
    After some transformations.
    """
    def __init__(self,
                 root_dir_path):
        if not os.path.isdir(root_dir_path):
            raise IOError("Dataset folder {} not found!".format(
                          root_dir_path))
        self.root_path = root_dir_path
        self.data_paths = glob2.glob(self.root_path)
        self.n_samples = len(self.data_paths)

    def __len__(self):
        return self.n_samples

    def __getitem__(self, idx):
        sample = self.data_paths[idx]
        return sample


if __name__ == "__main__":
    root_dir = '/home/thymios/data'
    data_parametered_name = 'timit_256_64_128_2_m_-1taus1'
    partition = 'train'
    dataset_path = os.path.join(root_dir,
                                data_parametered_name,
                                partition)
    train_data = PytorchMixtureDataset(dataset_path)

    print(train_data)
