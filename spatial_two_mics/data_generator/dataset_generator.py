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
from librosa.core import stft

root_dir = os.path.join(
    os.path.dirname(os.path.realpath(__file__)),
    '../../')
sys.path.insert(0, root_dir)

import spatial_two_mics.data_loaders.timit as timit_loader
import spatial_two_mics.data_generator.source_position_generator as \
    positions_generator


class ArtificialDatasetCreator(object):
    """
    This is a general compatible class for creating Artificial
    mixtures with positions of the different sources.
    """
    def __init__(self,
                 audio_dataset_name="timit"):
        if audio_dataset_name.lower() == "timit":
            self.data_loader = timit_loader.TimitLoader()
            self.fs = 16000
        else:
            raise NotImplementedError("Dataset Loader: {} is not yet "
                  "implemented.".format(audio_dataset_name))


class RandomCombinations(ArtificialDatasetCreator):
    def __init__(self,
                 audio_dataset_name="timit",
                 genders_mixtures=None,
                 excluded_speakers=None,
                 subset_of_speakers='train',
                 min_duration=2.0):

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

        self.min_samples = int(min_duration * self.fs)

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

            # not a valid gender
            if not all(good_gender_mix):
                continue

            # we do not want the same speaker twice
            speaker_set = set([x['speaker_id'] for x in possible_comb])
            if len(speaker_set) < len(possible_comb):
                continue

            # check whether all the signals have the appropriate
            # duration
            signals = [(len(self.get_wav(speakers_dic, source_info))
                        >= self.min_samples)
                       for source_info in possible_comb]
            if not all(signals):
                continue

            valid_mixtures.append(possible_comb)

        return valid_mixtures

    @staticmethod
    def get_wav(speakers_dic,
                source_info):
        return speakers_dic[source_info['speaker_id']][
               'sentences'][source_info['sentence_id']]['wav']

    def construct_delayed_signals(self,
                                  signals,
                                  taus,
                                  force_all_signals_delay=False):
        """!
        This function might extend to any real delay by interpolation
        of the source signals
        """

        # naive way in order to force a delay for DUET algorithm
        delays = []
        if force_all_signals_delay:
            delays = []
            for tau in taus:
                if tau >= 0:
                    delays.append(1)
                else:
                    delays.append(-1)

        delayed_signals = []
        for i, delay in enumerate(delays):
            new_signal = np.roll(signals[i], -delay)
            if delay > 0:
                new_signal[-delay:] = 0.
            elif delay < 0:
                new_signal[:-delay] = 0.
            delayed_signals.append(new_signal)

        return delayed_signals

    def construct_mixture_signals(self,
                                  source_signals,
                                  positions,
                                  force_all_signals_delay=False):
        """!
        This function constructs the mixture for each mic (m1,
        m2) in the following way:
        m1(t) = a1*s1(t) + ... + an*sn(t)
        m2(t) = a1*s1(t+d1) + ... + an*sn(t+dn)

        by also cutting them off to self.min_samples
        """

        cropped_signals = [s[:self.min_samples]
                           for s in source_signals]
        delayed_signals = self.construct_delayed_signals(
                               cropped_signals,
                               positions['taus'],
                               force_all_signals_delay=force_all_signals_delay)

        m1 = sum([positions['amplitudes'][i]*cropped_signals[i]
                  for i in np.arange(len(cropped_signals))])

        m2 = sum([positions['amplitudes'][i] * delayed_signals[i]
                  for i in np.arange(len(delayed_signals))])

        sources_spectra = [stft(s, n_fft=1024, win_length=320)
                           for s in cropped_signals]

        delayed_sources_spectra = [stft(s, n_fft=1024, win_length=320)
                                   for s in delayed_signals]

        m1_tf = stft(m1, n_fft=1024, win_length=320)
        m2_tf = stft(m2, n_fft=1024, win_length=320)

        mixture_info = {
            'm1_raw': m1,
            'm2_raw': m2,
            'm1_tf': m1_tf,
            'm2_tf': m2_tf,
            'sources_raw': cropped_signals,
            'sources_tf': sources_spectra,
            'delayed_sources_raw': delayed_signals,
            'delayed_sources_tf': delayed_sources_spectra,
        }

        return mixture_info

    def acquire_mixture_information(self,
                                    speakers_dic,
                                    combination_info,
                                    positioner,
                                    force_all_signals_delay=False):
        """! The whole processing for getting the mixture signals for
        the two mics and the positions is done here.

        :param positioner should be able to return:
               'amplitudes': array([0.28292362, 0.08583346, 0.63124292]),
               'd_thetas': array([1.37373734, 1.76785531]),
               'distances': {'m1m1': 0.0,
                             'm1m2': 0.03,
                             'm1s1': 3.015, ...
                             's3s3': 0.0},
               'taus': array([ 1.456332, -1.243543,  0]),
               'thetas': array([0.        , 1.37373734, 3.14159265]),
               'xy_positons': array([[ 3.00000000e+00, 0.00000000e+00],
                   [ 5.87358252e-01,  2.94193988e+00],
                   [-3.00000000e+00,  3.67394040e-16]])}

        :param speakers_dic should be able to return a dic like this:
                'speaker_id_i': {
                    'dialect': which dialect the speaker belongs to,
                    'gender': f or m,
                    'sentences': {
                        'sentence_id_j': {
                            'wav': wav_on_a_numpy_matrix,
                            'sr': Fs in Hz integer,
                            'path': PAth of the located wav
                        }
                    }
                }

        :param combination_info should be in the following format:
           [{'gender': 'm', 'sentence_id': 'sx298', 'speaker_id': 'mctt0'},
            {'gender': 'm', 'sentence_id': 'sx364', 'speaker_id': 'mrjs0'},
           {'gender': 'f', 'sentence_id': 'sx369', 'speaker_id': 'fgjd0'}]

        """
        n_sources = len(combination_info)
        positions = positioner.get_sources_locations(n_sources)

        source_signals = [self.get_wav(speakers_dic, source_info)
                          for source_info in combination_info]

        mixture_info = self.construct_mixture_signals(
                            source_signals,
                            positions,
                            force_all_signals_delay=force_all_signals_delay)

        return mixture_info

    def get_mixture_combinations(self,
                                 n_sources_in_mix=2,
                                 n_mixtures=0,
                                 force_all_signals_delay=False):
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

        random_positioner = positions_generator.RandomCirclePositioner()
        mixtures = [self.acquire_mixture_information(
                         speakers_dic,
                         combination,
                         random_positioner,
                         force_all_signals_delay=force_all_signals_delay)
                    for combination in valid_combinations]

        print(len(mixtures))
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
                           n_mixtures=1000,
                           force_all_signals_delay=True)


if __name__ == "__main__":
    example_of_usage()

