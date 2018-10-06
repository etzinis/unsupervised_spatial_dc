"""!
@brief A dataset creation which is used in order to combine the
mixtures form the dataset and also store them inside a specified folder

@author Efthymios Tzinis {etzinis2@illinois.edu}
@copyright University of Illinois at Urbana Champaign
"""

import argparse
import os
import sys
import numpy as np
from random import shuffle
from pprint import pprint
from sklearn.externals import joblib

root_dir = os.path.join(
    os.path.dirname(os.path.realpath(__file__)),
    '../../')
sys.path.insert(0, root_dir)

import spatial_two_mics.data_loaders.timit as timit_loader
import spatial_two_mics.data_generator.source_position_generator as \
    positions_generator
import spatial_two_mics.labels_inference.tf_label_estimator as \
    mask_estimator
import spatial_two_mics.utils.audio_mixture_constructor as \
            mix_constructor
import spatial_two_mics.utils.progress_display as progress_display


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
                 mixture_distribution=None,
                 create_val_set=False,
                 subset_of_speakers='train',
                 min_duration=2.0):

        super(RandomCombinations,
              self).__init__(audio_dataset_name=audio_dataset_name)

        self.mixture_distribution = mixture_distribution
        self.audio_dataset_name = audio_dataset_name
        self.data_dic = self.data_loader.load()
        self.subset_of_speakers = subset_of_speakers

        self.genders_mixtures = genders_mixtures
        valid_genders = [(g in ['f', 'm'])
                         for g in self.genders_mixtures]
        assert valid_genders, ('Valid genders for mixtures are f and m')

        self.used_speakers = self.get_available_speakers(
                                  subset_of_speakers)
        print("All Available Speakers are {}".format(
            len(self.used_speakers)))

        if create_val_set:
            n_available = len(self.used_speakers)
            self.val_speakers = np.random.choice(self.used_speakers,
                                                 int(n_available/2),
                                                 replace=False)
        else:
            self.val_speakers = []

        self.used_speakers = [s for s in self.used_speakers
                              if s not in self.val_speakers]

        self.min_samples = int(min_duration * self.fs)

    def get_available_speakers(self,
                               subset_of_speakers):
        try:
            available_speakers = sorted(list(self.data_dic[
                                                 subset_of_speakers].keys()))
        except KeyError:
            print("Subset: {} not available".format(subset_of_speakers))
            raise KeyError

        valid_speakers = []
        for speaker in available_speakers:

            if ((self.data_dic[subset_of_speakers][speaker]['gender']
                 in self.genders_mixtures)):
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

    @staticmethod
    def get_wav_path(speakers_dic,
                source_info):
        return speakers_dic[source_info['speaker_id']][
            'sentences'][source_info['sentence_id']]['path']

    def construct_mixture_info(self,
                               speakers_dic,
                               combination_info,
                               positions):
        """
        :param positions should be able to return:
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

        :return condensed mixture information block:
        {
            'postions':postions (argument)
            'sources_ids':
            [       {
                        'gender': combination_info.gender
                        'sentence_id': combination_info.sentence_id
                        'speaker_id': combination_info.speaker_id
                        'wav_path': the wav_path for the file
                    } ... ]
        }
        """

        new_combs_info = combination_info.copy()

        for comb in new_combs_info:
            comb.update({'wav_path':
                         self.get_wav_path(speakers_dic,
                                           comb)})

        return {'positions': positions,
                'sources_ids': new_combs_info}

    def gather_mixtures_information(self,
                                    speakers,
                                    n_sources_in_mix=2,
                                    n_mixtures=0):
        """
        speakers_dic should be able to return a dic like this:
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

        combination_info should be in the following format:
           [{'gender': 'm', 'sentence_id': 'sx298', 'speaker_id': 'mctt0'},
            {'gender': 'm', 'sentence_id': 'sx364', 'speaker_id': 'mrjs0'},
           {'gender': 'f', 'sentence_id': 'sx369', 'speaker_id': 'fgjd0'}]

        """
        speakers_dic = self.data_dic[self.subset_of_speakers]

        possible_sources = []
        for speaker in speakers:
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

        mixtures_info = [self.construct_mixture_info(
            speakers_dic,
            combination,
            random_positioner.get_sources_locations(len(
                combination)))
            for combination in valid_combinations]

        return mixtures_info

    def update_label_masks_and_info(self,
                                    mixture_info,
                                    mixture_creator=None,
                                    ground_truth_estimator=None,
                                    soft_label_estimator=None,
                                    output_dir=None):
        name = [s_id['speaker_id'] + '-' + s_id['sentence_id']
                for s_id in mixture_info['sources_ids']]
        name = '_'.join(name)
        data = {}

        tf_mixture = mixture_creator.construct_mixture(mixture_info)
        gt_mask = ground_truth_estimator.infer_mixture_labels(tf_mixture)
        data['ground_truth_mask'] = gt_mask
        data['real_tfs'] = np.real(tf_mixture['m1_tf'])
        data['imag_tfs'] = np.imag(tf_mixture['m1_tf'])
        data['wavs'] = tf_mixture['sources_raw']
        if soft_label_estimator is not None:
            duet_mask = soft_label_estimator.infer_mixture_labels(
                                             tf_mixture)
            data['soft_labeled_mask'] = duet_mask


        folder_path = os.path.join(output_dir, name)
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)
        # now just go over all the appropriate elements of mix info
        # dictionary and save them in separate files according to
        # their need
        for k, v in data.items():
            file_path = os.path.join(folder_path, k)
            joblib.dump(v, file_path, compress=0)
        return 1

    def combination_process_wrapper(self,
                                    output_dir=None,
                                    ground_truth_estimator=None,
                                    mixture_creator=None,
                                    soft_label_estimator=None):
        return lambda mix_info: \
               self.update_label_masks_and_info(
                    mix_info,
                    output_dir=output_dir,
                    ground_truth_estimator=ground_truth_estimator,
                    mixture_creator=mixture_creator,
                    soft_label_estimator=soft_label_estimator)

    def create_dataset_name(self,
                            n_samples=None,
                            n_sources=None,
                            force_delays=None):
        dataset_name = '{}_{}_{}_{}_{}'.format(
            self.audio_dataset_name,
            '_'.join(map(str, n_samples)),
            n_sources,
            ''.join(sorted(self.genders_mixtures)),
            'taus'.join(map(str, force_delays)))
        return dataset_name

    def get_mixture_combinations(self,
                                 speakers,
                                 output_dir=None,
                                 n_sources_in_mix=2,
                                 n_mixtures=0,
                                 force_delays=None,
                                 get_only_ground_truth=False):

        mixtures_info = self.gather_mixtures_information(
                        speakers,
                        n_sources_in_mix=n_sources_in_mix,
                        n_mixtures=n_mixtures)

        print("Created the combinations of all the speakers and "
              "ready to process each mixture separately!")

        mixture_creator = mix_constructor.AudioMixtureConstructor(
            n_fft=512, win_len=512, hop_len=128, mixture_duration=2.0,
            force_delays=force_delays)

        gt_estimator = mask_estimator.TFMaskEstimator(
                       inference_method='Ground_truth')

        if get_only_ground_truth:
            duet_estimator = None
        else:
            duet_estimator = mask_estimator.TFMaskEstimator(
                             inference_method='duet_Kmeans')

        mix_info_func = self.combination_process_wrapper(
                             output_dir=output_dir,
                             ground_truth_estimator=gt_estimator,
                             mixture_creator=mixture_creator,
                             soft_label_estimator=duet_estimator)
        n_mixes = str(len(mixtures_info))
        mixtures_info = progress_display.progress_bar_wrapper(
                                         mix_info_func,
                                         mixtures_info,
                                         message='Creating '
                                                  +n_mixes+
                                                  ' Mixtures...')

        print("Successfully stored {} mixtures".format(
              sum(mixtures_info)))

    def create_and_store_all_mixtures(self,
                                      n_sources_in_mix=2,
                                      n_mixtures=0,
                                      force_delays=None,
                                      get_only_ground_truth=False,
                                      output_dir=None,
                                      selected_partition=None):

        speakers = self.used_speakers
        if selected_partition is None:
            selected_partition = self.subset_of_speakers
        else:
            if selected_partition == 'val':
                speakers = self.val_speakers
            elif selected_partition == 'test':
                speakers = self.used_speakers
            else:
                raise ValueError("No valid partition named: {}".format(
                      selected_partition))

        storage_path = os.path.join(output_dir,
                                    self.create_dataset_name(
                                    n_samples=self.mixture_distribution,
                                    n_sources=n_sources_in_mix,
                                    force_delays=force_delays),
                                    selected_partition)

        self.get_mixture_combinations(
        speakers,
        n_sources_in_mix=n_sources_in_mix,
        n_mixtures=n_mixtures,
        force_delays=force_delays,
        get_only_ground_truth=
        get_only_ground_truth,
        output_dir=storage_path)


def generate_dataset(args):
    n_train, n_test, n_val = args.n_samples

    timit_mixture_creator = RandomCombinations(
        audio_dataset_name=args.dataset,
        genders_mixtures=args.genders,
        mixture_distribution=args.n_samples,
        subset_of_speakers='train',
        create_val_set=False)

    timit_mixture_creator.create_and_store_all_mixtures(
                            n_sources_in_mix=args.n_sources,
                            n_mixtures=n_train,
                            output_dir=args.output_path,
                            force_delays=args.force_delays)

    timit_mixture_creator = RandomCombinations(
        audio_dataset_name=args.dataset,
        genders_mixtures=args.genders,
        mixture_distribution=args.n_samples,
        subset_of_speakers='test',
        create_val_set=args.val_set)

    timit_mixture_creator.create_and_store_all_mixtures(
        n_sources_in_mix=args.n_sources,
        n_mixtures=n_test,
        output_dir=args.output_path,
        selected_partition='test',
        force_delays=args.force_delays)

    timit_mixture_creator.create_and_store_all_mixtures(
        n_sources_in_mix=args.n_sources,
        n_mixtures=n_val,
        selected_partition='val',
        output_dir=args.output_path,
        force_delays=args.force_delays)


def get_args():
    """! Command line parser """
    parser = argparse.ArgumentParser(description='Mixture dataset '
                                                 'creator')
    parser.add_argument("--dataset", type=str,
                        help="Dataset name", default="timit")
    parser.add_argument("--n_sources", type=int,
                        help="How many sources in each mix", default=2)
    parser.add_argument("--n_samples", type=int, nargs='+',
                        help="How many samples do u want to be "
                             "created",
                        default=[1,1,1])
    parser.add_argument("--genders", type=str, nargs='+',
                        help="Genders that will correspond to the "
                             "genders in the mixtures",
                        default=['m', 'f'])
    parser.add_argument("-o", "--output_path", type=str,
                        help="""The path that the resulting dataset 
                        would be stored. If the folder does not 
                        exist it will be created as well as its 
                        child folders train or test and val if it is 
                        selected""",
                        required=True)
    parser.add_argument("-f", "--force_delays", nargs='+', type=int,
                        help="""Whether you want to force integer 
                        delays of +- 1 in the sources e.g.""",
                        default=None)
    parser.add_argument('--val_set', action="store_true",
                        help='Force to create a separate val folder '
                             'with the same amount of the mixtures as '
                             'the initial test/train folder but using '
                             'half of the available speakers')
    return parser.parse_args()


if __name__ == "__main__":
    args = get_args()
    generate_dataset(args)

