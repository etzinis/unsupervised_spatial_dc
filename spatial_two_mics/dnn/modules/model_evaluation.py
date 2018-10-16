"""!
@brief For a specific dataset just apply the saved models on a
specific dataset and save the results

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
import torch
import itertools
import pandas as pd
from progress.bar import ChargingBar

root_dir = os.path.join(
    os.path.dirname(os.path.realpath(__file__)),
    '../../../')
sys.path.insert(0, root_dir)
import spatial_two_mics.dnn.utils.fast_dataset_v3 as data_loader
import spatial_two_mics.dnn.evaluation.naive_evaluation_numpy as np_eval
import spatial_two_mics.dnn.utils.model_logger as model_logger
from spatial_two_mics.config import RESULTS_DIR
import spatial_two_mics.dnn.evaluation.naive_evaluation_numpy as \
    numpy_eval
from sklearn.cluster import KMeans


def eval(dataset_gen,
         model_path,
         dataset_dir,
         partition,
         n_sources,
         n_batches,
         n_jobs):

    model_name = os.path.basename(model_path)

    eval_dic = {'sdr': [], 'sir': [], 'sar': []}

    model, optimizer, _, _, args, mean_tr, std_tr = \
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

                embedding_labels = np.array(k_means_obj.fit_predict(
                    embedding_features))

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
    mean_std_dic = {}
    for k, v in eval_dic.items():
        mean_std_dic[k+"_mean"] = np.mean(np.array(v))
        mean_std_dic[k+"_std"] = np.std(np.array(v))
    mean_std_dic['hidden_size'] = args.hidden_size
    mean_std_dic['num_layers'] = args.n_layers
    mean_std_dic['embedding_depth'] = args.embedding_depth
    mean_std_dic['dropout'] = str(args.dropout)
    mean_std_dic['lr'] = args.learning_rate

    return model_name, mean_std_dic


def evaluate_models(pretrained_models,
                    dataset_folder,
                    n_jobs=1,
                    get_top=None):
    visible_cuda_ids = ','.join(map(str, args.cuda_available_devices))
    os.environ["CUDA_VISIBLE_DEVICES"] = visible_cuda_ids

    (dataset_dir, partition) = (os.path.dirname(dataset_folder),
                                os.path.basename(dataset_folder))

    default_bs = 4
    if get_top is None:
        loading_bs = default_bs
    else:
        loading_bs = min(default_bs, get_top)

    print("Initializing the data loader for the dataset...")
    val_generator, n_val_batches, n_val_sources = \
        data_loader.get_data_generator(dataset_dir,
                                       partition=partition,
                                       get_top=get_top,
                                       num_workers=n_jobs,
                                       return_stats=False,
                                       return_n_batches=True,
                                       return_n_sources=True,
                                       batch_size=loading_bs)

    eval_results = {}
    for model_path in pretrained_models:

        try:
            test_on = os.path.basename(dataset_dir) + '_' + partition
            train_on = os.path.basename(os.path.dirname(model_path))
            folder_name = os.path.join(RESULTS_DIR,
                                       'test_on_' + test_on)
            if not os.path.exists(folder_name):
                os.makedirs(folder_name)

            # if the df already exists then do not make the evaluation
            df_path = os.path.join(folder_name,
                                   'train_on_' + train_on + '.csv')
            if os.path.lexists(df_path):
                df = pd.read_csv(df_path)
                df.set_index("Unnamed: 0", drop=True, inplace=True)
                eval_results = df.to_dict(orient='index')

                if os.path.basename(model_path) in eval_results.keys():
                    continue

            model_name, res = eval(val_generator,
                                   model_path,
                                   dataset_dir,
                                   partition,
                                   n_val_sources,
                                   n_val_batches,
                                   n_jobs)

            eval_results[model_name] = res

            df = pd.DataFrame(eval_results).T
            df = df.sort_values(['sdr_mean'], ascending=False)
            df.to_csv(df_path)
        except Exception as e:
            print(e)

    return df


def get_args():
    """! Command line parser for computing the evaluation for
    specific datasets"""
    parser = argparse.ArgumentParser(description='Evaluating'
             ' stored models for a specific dataset')
    parser.add_argument("-d", "--dataset_to_test", type=str,
                        help="Dataset path you want to evaluate",
                        default=None)
    parser.add_argument("-m", "--pretrained_models", type=str,
                        nargs='+',
                        help="Paths of pretrained models that you "
                             "need to test on this dataset",
                        default=[])
    parser.add_argument("--n_jobs", type=int,
                        help="Number of parallel spawning jobs",
                        default=1)
    parser.add_argument("--n_eval", type=int,
                        help="""Reduce the number of evaluation 
                            samples to this number.""", default=None)
    parser.add_argument("-cad", "--cuda_available_devices", type=int,
                        nargs="+",
                        help="""A list of Cuda IDs that would be 
                            available for running this experiment""",
                        default=[0])
    return parser.parse_args()


if __name__ == "__main__":
    args = get_args()
    df_results = evaluate_models(args.pretrained_models,
                                 args.dataset_to_test,
                                 n_jobs=args.n_jobs,
                                 get_top=args.n_eval)

    pd.set_option('display.expand_frame_repr', False)
    print(df_results.sort_values(['sdr_mean'], ascending=False))
