"""!
@brief A dataset creation which is used in order to combine the
mixtures form the dataset and also store them inside a specified folder

@author Efthymios Tzinis {etzinis2@illinois.edu}
@copyright University of Illinois at Urbana Champaign
"""

import argparse
import os
import sys
from pprint import pprint
from sklearn.externals import joblib

root_dir = os.path.join(
    os.path.dirname(os.path.realpath(__file__)),
    '../../')
sys.path.insert(0, root_dir)

import spatial_two_mics.data_generator.dataset_generator as generator


def create_dataset_name(args):
    dataset_name = '{}_{}_{}_{}_{}'.format(
                    args.dataset,
                    '_'.join(map(str, args.n_samples)),
                    args.n_sources,
                    ''.join(sorted(args.genders)),
                    'taus'.join(map(str,  args.force_delays)))
    return dataset_name


def get_mixture_name_and_data_to_save(mix_info):
    name = [s_id['speaker_id']+'-'+s_id['sentence_id']
            for s_id in mix_info['sources_ids']]
    name = '_'.join(name)

    data = {
        'amplitudes': mix_info['positions']['amplitudes'],
        'wav_paths': [s_id['wav_path']
                      for s_id in mix_info['sources_ids']],
        'ground_truth_mask': mix_info['ground_truth_mask']
    }

    if 'soft_labeled_mask' in mix_info:
        data['soft_labeled_mask'] = mix_info['soft_labeled_mask']

    return name, data

def time_loading_comparison(data, f_path):
    import _pickle as cPickle
    from sklearn.externals import joblib
    import time

    joblib.dump(data, f_path)
    before = time.time()
    tempos = joblib.load(f_path)
    now = time.time()
    jlib_time = now - before

    cPickle.dump(data, open(f_path, 'wb'))
    before = time.time()
    tempos = cPickle.load(open(f_path, 'rb'))
    now = time.time()
    pickle_time = now - before

    return jlib_time, pickle_time


def store_dataset(dataset_dic, args):

    dataset_name = create_dataset_name(args)

    dataset_path = os.path.join(args.output_path, dataset_name)
    if not os.path.exists(dataset_path):
        os.makedirs(dataset_path)

    for subf, mixtures_info in dataset_dic.items():
        subf_path = os.path.join(dataset_path, subf)
        if not os.path.exists(subf_path):
            os.makedirs(subf_path)

        for mix_info in mixtures_info:
            name, data = get_mixture_name_and_data_to_save(mix_info)
            f_path = os.path.join(subf_path, name)
            joblib.dump(data, f_path, compress=3)


def generate_dataset(args):
    n_train, n_test, n_val = args.n_samples
    timit_mixture_creator = generator.RandomCombinations(
        audio_dataset_name=args.dataset,
        genders_mixtures=args.genders,
        subset_of_speakers='train',
        create_val_set=False)

    dataset_dic = timit_mixture_creator.get_all_mixture_sets(
        n_sources_in_mix=args.n_sources,
        n_mixtures=n_train,
        force_delays=args.force_delays)

    timit_mixture_creator = generator.RandomCombinations(
        audio_dataset_name=args.dataset,
        genders_mixtures=args.genders,
        subset_of_speakers='test',
        create_val_set=True)

    test_val_dic = timit_mixture_creator.get_all_mixture_sets(
        n_sources_in_mix=args.n_sources,
        n_mixtures=max(n_test, n_val),
        force_delays=args.force_delays)

    dataset_dic.update(test_val_dic)
    return dataset_dic


def create_and_store_dataset(args):
    dataset_dic = generate_dataset(args)
    store_dataset(dataset_dic, args)


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
                             "created for train test val",
                        default=10)
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
    create_and_store_dataset(args)

