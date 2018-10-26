"""!
@brief sourse separation performance all eval values

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
import joblib

root_dir = os.path.join(
    os.path.dirname(os.path.realpath(__file__)),
    '../../../')
sys.path.insert(0, root_dir)
import spatial_two_mics.dnn.utils.fast_dataset_v3 as data_loader
import spatial_two_mics.dnn.evaluation.naive_evaluation_numpy as np_eval
from spatial_two_mics.config import FINAL_RESULTS_DIR


def eval(data_generator,
         dataset_path):

    data_dir = os.path.dirname(dataset_path)
    info = os.path.basename(data_dir)
    n_sources = int(info.split('_')[4])

    eval_dic = {'sdr': [], 'sir': [], 'sar': []}

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

            eval_dic['sdr'].append(sdr)
            eval_dic['sir'].append(sir)
            eval_dic['sar'].append(sar)

    # return all values
    result_dic = {}
    for k, v in eval_dic.items():
        result_dic[k] = np.array(v)

    return result_dic


def evaluate_labels(dataset_folder,
                    n_jobs=1,
                    get_top=None):
    (dataset_dir, partition) = (os.path.dirname(dataset_folder),
                                os.path.basename(dataset_folder))

    assert partition == 'test' or partition == 'val', '' \
           'All selected dataset folder to be evaluated have either ' \
           'to be test or val folder from a certain dataset!'

    eval_results={}
    for eval_labels in ['duet', 'ground_truth']:
        val_generator, n_val_batches = \
            data_loader.get_data_generator(
                            dataset_dir, partition=partition,
                            get_top=get_top, num_workers=1,
                            return_stats=False, labels_mask=eval_labels,
                            return_n_batches=True,
                            only_mask_evaluation=True)

        eval_results[eval_labels] = eval(val_generator,
                                         os.path.join(dataset_dir,
                                                      partition))

    return eval_results


def get_args():
    """! Command line parser for computing the evaluation for
    specific datasets"""
    parser = argparse.ArgumentParser(description='Evaluating'
             ' groundtruth or duet labels for datasets folders')
    parser.add_argument("-i", "--dataset_folders", type=str, nargs='+',
                        help="Dataset paths you want to evaluate",
                        default=[])
    parser.add_argument("--n_jobs", type=int,
                        help="Number of parallel spawinign jobs",
                        default=1)
    parser.add_argument("--n_eval", type=int,
                        help="""Reduce the number of evaluation 
                            samples to this number.""", default=None)
    return parser.parse_args()


if __name__ == "__main__":
    args = get_args()


    for dataset_folder in args.dataset_folders:
        (dataset_dir, partition) = (os.path.dirname(dataset_folder),
                                    os.path.basename(dataset_folder))

        eval_results = evaluate_labels(dataset_folder,
                                       n_jobs=args.n_jobs,
                                       get_top=args.n_eval)

        pprint(eval_results)

        test_on = os.path.basename(dataset_dir) + '_' + partition
        save_folder_name = os.path.join(FINAL_RESULTS_DIR,
                                        'test_on_' + test_on)
        if not os.path.exists(save_folder_name):
            os.makedirs(save_folder_name)

        for labels, metrics in eval_results.items():
            file_path = os.path.join(save_folder_name,
                                     labels + '_mask_metrics.gz')

            joblib.dump(metrics, file_path)
