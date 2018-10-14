"""!
@brief Create datasets for the experiments by individually assign
them as jobs in different processors

@author Efthymios Tzinis {etzinis2@illinois.edu}
@copyright University of illinois at Urbana Champaign
"""

import argparse
import os
import sys
import itertools
import copy
from pprint import pprint
root_dir = os.path.join(
    os.path.dirname(os.path.realpath(__file__)),
    '../../')
sys.path.insert(0, root_dir)
from joblib import Parallel, delayed
import spatial_two_mics.data_generator.data_creator_and_storage_v2 as\
    dataset_generator


def generate_one_dataset_wrapper(this_dataset_args):
    dataset_generator.generate_dataset(this_dataset_args)
    return 1


def generate_datasets(args):
    genders = list(map(list, args.genders))
    n_sources = args.n_sources

    dataset_combinations = list(itertools.product(*[genders,
                                                    n_sources]))

    specific_args = []
    for (gndrs, sources) in dataset_combinations:
        this_args = copy.deepcopy(args)
        this_args.n_sources = sources
        this_args.genders = gndrs
        del this_args.n_jobs
        specific_args.append(this_args)

    pprint(specific_args)

    created_datasets = Parallel(n_jobs=args.n_jobs)(
                       [delayed(generate_one_dataset_wrapper)(this_args)
                        for this_args in specific_args])

    print("Successfully created: {} datasets".format(
          sum(created_datasets)))

    return True


def get_args():
    """! Command line parser """
    parser = argparse.ArgumentParser(description='Parallel Mixture '
                                                 'datasets creator')
    parser.add_argument("--dataset", type=str,
                        help="Dataset name", default="timit")
    parser.add_argument("--n_sources", type=int, nargs='+',
                        help="How many sources in each mix", default=2)
    parser.add_argument("--n_samples", type=int, nargs='+',
                        help="How many samples do u want to be "
                             "created",
                        default=[1, 1, 1])
    parser.add_argument("--genders", type=str, nargs='+',
                        help="Genders that will correspond to the "
                             "genders in the mixtures",
                        default=['m'], choices=['m', 'f', 'fm', 'mf'])
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
    parser.add_argument("--n_jobs", type=int,
                        help="Number of parallel spawning jobs",
                        default=1)
    return parser.parse_args()


if __name__ == "__main__":
    args = get_args()
    generate_datasets(args)