"""!
@brief For a specific dataset just find all the groundtruth
evaluation when applying either a duet or a ground truth labeled mask
for source separation

@author Efthymios Tzinis {etzinis2@illinois.edu}
@copyright University of illinois at Urbana Champaign
"""

import argparse
import os
import sys
import numpy as np
from pprint import pprint
from joblib import Parallel, delayed
from tqdm import tqdm
import itertools
import pandas as pd

root_dir = os.path.join(
    os.path.dirname(os.path.realpath(__file__)),
    '../../../')
sys.path.insert(0, root_dir)
import spatial_two_mics.dnn.utils.fast_dataset_v3 as data_loader
import spatial_two_mics.dnn.evaluation.naive_evaluation_numpy as np_eval


def eval(data_generator,
         n_batches,
         dataset_path):

    data_dir = os.path.dirname(dataset_path)
    info = os.path.basename(data_dir)
    n_sources = int(info.split('_')[4])

    eval_dic = {'sdr': 0., 'sir': 0., 'sar': 0.}

    for batch_data in data_generator:
        abs_tfs, masks, wavs_lists, real_tfs, imag_tfs = batch_data

        for b in np.arange(abs_tfs.size(0)):
            embedding_labels = masks[b].data.numpy()

            sdr, sir, sar = np_eval.naive_cpu_bss_eval(
                embedding_labels,
                real_tfs[b].data.numpy(),
                imag_tfs[b].data.numpy(),
                wavs_lists[b].data.numpy(),
                n_sources,
                batch_index=b)

            eval_dic['sdr'] += sdr / (1. * n_batches * abs_tfs.size(0))
            eval_dic['sir'] += sir / (1. * n_batches * abs_tfs.size(0))
            eval_dic['sar'] += sar / (1. * n_batches * abs_tfs.size(0))

            print(sdr, sir, sar)
    input("Check")

    return dataset_path, eval_dic


def eval_wrapper():
    return lambda data_generator, n_batches, dataset_path: eval(
                  data_generator, n_batches, dataset_path)


def evaluate_labels(dataset_folders,
                    eval_labels='duet',
                    n_jobs=1,
                    get_top=None):

    # n_workers = min(len(dataset_folders), n_jobs)
    n_workers = n_jobs
    dirs_and_parts = [(os.path.dirname(f), os.path.basename(f))
                      for f in dataset_folders]

    assert all([partition == 'test' or partition == 'val'
                for (_, partition) in dirs_and_parts]), '' \
           'All selected dataset folder to be evaluated have either ' \
           'to be test or val folder from a certain dataset!'

    print("Initializing the data loaders for all the datasets...")
    datasets_loaders = [data_loader.get_data_generator(
                        dataset_dir, partition=partition,
                        get_top=get_top, num_workers=1, return_stats=False,
                        labels_mask=eval_labels, return_n_batches=True)
                        for (dataset_dir, partition) in dirs_and_parts]

    data_info = [list(itertools.chain.from_iterable(info_lists))
                 for info_lists in zip(datasets_loaders, dirs_and_parts)]

    eval_results = [eval(data_loader,
                         n_batches,
                         os.path.join(data_dir, partition))
                    for (data_loader, n_batches, data_dir, partition)
                    in data_info]

    eval_results = Parallel(n_jobs=n_jobs)(
                   [delayed(eval)(data_loader,
                                  n_batches,
                                  os.path.join(data_dir, partition))
                   for (data_loader, n_batches, data_dir, partition)
                   in tqdm(data_info)])

    return eval_results


def get_args():
    """! Command line parser for computing the evaluation for
    specific datasets"""
    parser = argparse.ArgumentParser(description='Evaluating'
             ' groundtruth or duet labels for datasets folders')
    parser.add_argument("-i", "--dataset_folders", type=str, nargs='+',
                        help="Dataset paths you want to evaluate",
                        default=[])
    parser.add_argument("-l", "--eval_labels", type=str,
                        help="Choose what labels do you want to use "
                             "for the evaluation",
                        default='duet', choices=['duet',
                                                 'ground_truth'])
    parser.add_argument("--n_jobs", type=int,
                        help="Number of parallel spawinign jobs",
                        default=1)
    parser.add_argument("--n_eval", type=int,
                        help="""Reduce the number of evaluation 
                            samples to this number.""", default=None)
    return parser.parse_args()


if __name__ == "__main__":
    args = get_args()
    eval_results = evaluate_labels(args.dataset_folders,
                                   eval_labels=args.eval_labels,
                                   n_jobs=args.n_jobs,
                                   get_top=args.n_eval)

    df = pd.DataFrame(dict([(os.path.basename(os.path.dirname(p)) +
                             '/' + os.path.basename(p), res)
                            for (p, res) in eval_results])).T
    pd.set_option('display.expand_frame_repr', False)
    print(df)
