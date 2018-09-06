"""!
@brief A dataset creation which is used in order to combine the
mixtures form the dataset and also store them inside a specified folder

@author Efthymios Tzinis {etzinis2@illinois.edu}
@copyright University of illinois at Urbana Champaign
"""

import os
import sys
import numpy as np
from random import shuffle
from pprint import pprint
from torch.utils.data import Dataset, DataLoader

root_dir = os.path.join(
    os.path.dirname(os.path.realpath(__file__)),
    '../../')
sys.path.insert(0, root_dir)

import spatial_two_mics.data_loaders.timit as timit_loader


class ArtificialDatasetCreator(object):
    """
    This is a general compatible class for creating Artificial
    mixtures with positions of the different sources.
    """
    def __init__(self,
                 audio_dataset_name="timit"):
        if audio_dataset_name.lower() == "timit":
            self.data_loader = timit_loader.TimitLoader()
        else:
            raise NotImplementedError("Dataset Loader: {} is not yet "
                  "implemented.".format(audio_dataset_name))


class RandomCombinations(ArtificialDatasetCreator):
    def __init__(self,
                 audio_dataset_name="timit",
                 genders_mixtures=None,
                 excluded_speakers=None,
                 subset_of_speakers='train'):
        super(RandomCombinations,
              self).__init__(audio_dataset_name=audio_dataset_name)

        self.data_dic = self.data_loader.load()
        self.subset_of_speakers = subset_of_speakers

        if excluded_speakers is None:
            excluded_speakers = []

        self.genders_mixtures = genders_mixtures
        valid_genders = [(g in ['f', 'm'])
                         for g in self.genders_mixtures]
        assert valid_genders, ('Valid genders for mixtures are f and m')

        self.used_speakers = self.get_available_speakers(
                                  subset_of_speakers,
                                  excluded_speakers)

    def get_available_speakers(self,
                               subset_of_speakers,
                               excluded_speakers):
        try:
            available_speakers = sorted(list(self.data_dic[
                                                 subset_of_speakers].keys()))
        except KeyError:
            print("Subset: {} not available".format(subset_of_speakers))
            raise KeyError

        valid_speakers = []
        for speaker in available_speakers:

            if ((speaker not in excluded_speakers) and
                    (self.data_dic[subset_of_speakers][
                         speaker]['gender'] in self.genders_mixtures)):
                valid_speakers.append(speaker)

        return valid_speakers

    @staticmethod
    def random_combinations(iterable, r):
        iter_len = len(iterable)
        max_combs = 1
        for i in np.arange(r):
            max_combs *= (iter_len - i + 1) / (i + 1)

        already_seen = set()
        c = 0
        while c < max_combs:
            indexes = sorted(np.random.choice(iter_len, r))
            str_indexes = str(indexes)
            if str_indexes in already_seen:
                continue
            else:
                already_seen.add(str_indexes)

            c += 1
            yield [iterable[i] for i in indexes]

    def acquire_mixture_information(self,
                                    speaker_dic,
                                    combination_info):
        return None

    def get_only_valid_mixture_combinations(self,
                                            possible_sources,
                                            speakers_dic,
                                            n_mixed_sources=2,
                                            n_mixtures=0):
        mixtures_generator = self.random_combinations(possible_sources,
                                                      n_mixed_sources)

        if n_mixtures <= 0:
            print("All available mixtures that can be generated would "
                  " be: {}!".format(len(list(mixtures_generator))))
            print("Please Select a number of mixtures > 0")

        valid_mixtures = []

        while len(valid_mixtures) < n_mixtures:
            possible_comb = next(mixtures_generator)
            genders_in_mix = [x['gender'] for x in possible_comb]
            good_gender_mix = [g in genders_in_mix
                               for g in self.genders_mixtures]
            if not all(good_gender_mix):
                # not a valid gender
                continue

            valid_mixtures.append(self.acquire_mixture_information(
                speakers_dic,
                possible_comb))

        return valid_mixtures

    def get_mixture_combinations(self,
                                 n_sources_in_mix=2,
                                 n_mixtures=0):
        speakers_dic = self.data_dic[self.subset_of_speakers]
        possible_sources = []
        for speaker in self.used_speakers:
            sentences = list(speakers_dic[speaker]['sentences'].keys())
            gender = speakers_dic[speaker]['gender']
            possible_sources += [{'speaker_id': speaker,
                                  'gender': gender,
                                  'sentence_id': sentence}
                                 for sentence in sentences]

        shuffle(possible_sources)

        valid_combinations = self.get_only_valid_mixture_combinations(
            possible_sources,
            speakers_dic,
            n_mixed_sources=n_sources_in_mix,
            n_mixtures=n_mixtures)

        mixtures = []
        print(len(valid_combinations))
        input()
        return mixtures


def example_of_usage():
    """!
    Creates a list of mixtures in appropriate format with all the
    information that might be needed next"""

    timit_mixture_creator = RandomCombinations(
                            audio_dataset_name="timit",
                            genders_mixtures=['m', 'f'],
                            subset_of_speakers='test',
                            excluded_speakers=['mwew0'])

    mixture_combinations = timit_mixture_creator.get_mixture_combinations(
                           n_sources_in_mix=3,
                           n_mixtures=10000)


if __name__ == "__main__":
    example_of_usage()

