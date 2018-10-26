"""!
@brief Initial SDR all measurements and not only stat values

@author Efthymios Tzinis {etzinis2@illinois.edu}
@copyright University of illinois at Urbana Champaign
"""

import argparse
import os
import sys
import numpy as np
from pprint import pprint
import joblib
from sklearn.cluster import KMeans
from progress.bar import ChargingBar
import torch
import pandas as pd


root_dir = os.path.join(
    os.path.dirname(os.path.realpath(__file__)),
    '../../../')
sys.path.insert(0, root_dir)
import spatial_two_mics.dnn.utils.model_logger as model_logger
import spatial_two_mics.dnn.utils.fast_dataset_v3 as data_loader
import spatial_two_mics.dnn.evaluation.naive_evaluation_numpy as np_eval
from spatial_two_mics.config import *


def eval(dataset_gen,
         model_path,
         n_sources,
         n_batches,
         n_jobs):

    model_name = os.path.basename(model_path)

    eval_dic = {'sdr': [], 'sir': [], 'sar': []}

    model, optimizer, _, _, args, mean_tr, std_tr, training_labels = \
        model_logger.load_and_create_the_model(model_path)

    k_means_obj = KMeans(n_clusters=n_sources, n_jobs=n_jobs)

    model.eval()
    with torch.no_grad():
        bar = ChargingBar("Evaluating model {} ...".format(model_name),
                          max=n_batches)
        for batch_data in dataset_gen:
            abs_tfs, wavs_lists, real_tfs, imag_tfs = batch_data
            input_tfs = abs_tfs.cuda()
            # the input sequence is determined by time and not freqs
            # before: input_tfs = batch_size x (n_fft/2+1) x n_timesteps
            input_tfs = input_tfs.permute(0, 2, 1).contiguous()

            # normalize with mean and variance from the training dataset
            input_tfs -= mean_tr
            input_tfs /= std_tr

            vs = model(input_tfs)
            for b in np.arange(vs.size(0)):
                embedding_features = vs[b, :, :].data.cpu().numpy()

                z_embds = (embedding_features -
                           np.mean(embedding_features, axis=0)) / (
                           np.std(embedding_features, axis=0) + 10e-8)

                embedding_labels = np.array(k_means_obj.fit_predict(
                    z_embds))

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

            bar.next()
        bar.finish()

    # return both mean and std values
    result_dic = {}
    for k, v in eval_dic.items():
        result_dic[k] = np.array(v)

    return result_dic


def find_best_model_and_evaluate(args):

    visible_cuda_ids = ','.join(map(str, args.cuda_available_devices))
    os.environ["CUDA_VISIBLE_DEVICES"] = visible_cuda_ids

    for result_path in args.results_paths:
        (dataset_name,
         model_dataset) = (os.path.basename(os.path.dirname(
            result_path)).split(
            "test_on_")[1],
                           os.path.basename(
                               result_path).split(
                               "train_on_")[1].split(".csv")[0])

        partition = dataset_name.split('_')[-1]
        dataset_dirname = dataset_name.split('_' + partition)[0]

        print(dataset_dirname)

        df = pd.read_csv(result_path)

        mask_types2model_dir = {
            'duet': os.path.join(MODELS_DIR, model_dataset),
            'ground_truth': os.path.join(MODELS_GROUND_TRUTH,
                                         model_dataset),
            'raw_phase_diff': os.path.join(MODELS_RAW_PHASE_DIR,
                                           model_dataset)}

        for mask_type, saved_models_dir in mask_types2model_dir.items():
            mask_df = df.loc[df['training_labels'] == mask_type]
            mask_df = mask_df.sort_values(['sdr_mean'], ascending=False)
            mask_df.reset_index(drop=True, inplace=True)

            best_model_name = mask_df['Unnamed: 0'].loc[0]

            # construct model path
            best_model_p = os.path.join(saved_models_dir,
                                        best_model_name)

            if not os.path.exists(best_model_p):
                print(best_model_p)
                raise IOError("Model path not found!")

            test_dataset_dir = os.path.join(DATASETS_DIR,
                                            dataset_dirname)

            if not os.path.exists(test_dataset_dir):
                print(test_dataset_dir)
                raise IOError("Dataset path not found!")

            val_generator, n_val_batches, n_val_sources = \
                data_loader.get_data_generator(test_dataset_dir,
                                               partition=partition,
                                               get_top=args.n_eval,
                                               num_workers=args.n_jobs,
                                               return_stats=False,
                                               return_n_batches=True,
                                               return_n_sources=True,
                                               batch_size=32)

            res = eval(val_generator,
                       best_model_p,
                       n_val_sources,
                       n_val_batches,
                       args.n_jobs)

            test_on = os.path.basename(dataset_dirname) + '_' + partition
            save_folder_name = os.path.join(FINAL_RESULTS_DIR,
                                            'test_on_' + test_on)
            if not os.path.exists(save_folder_name):
                os.makedirs(save_folder_name)

            file_path = os.path.join(save_folder_name,
                        mask_type+'_deep_clustering_metrics.gz')

            pprint(res)

            joblib.dump(res, file_path)


def get_args():
    """! Command line parser for computing the evaluation for
    specific datasets"""
    parser = argparse.ArgumentParser(description='Evaluating'
             ' SDR SAR and SIR for datasets for the best models')
    parser.add_argument("-i", "--results_paths", type=str, nargs='+',
                        help="Results for datasets",
                        default=None)
    parser.add_argument("--n_jobs", type=int,
                        help="Number of parallel spawinign jobs",
                        default=1)
    parser.add_argument("-cad", "--cuda_available_devices", type=int,
                        nargs="+",
                        help="""A list of Cuda IDs that would be 
                                available for running this experiment""",
                        default=[2])
    parser.add_argument("--n_eval", type=int,
                        help="""Reduce the number of evaluation 
                            samples to this number.""", default=None)
    return parser.parse_args()


if __name__ == "__main__":
    args = get_args()
    find_best_model_and_evaluate(args)
