"""!
@brief Command line parser for experiments

@author Efthymios Tzinis {etzinis2@illinois.edu}
@copyright University of illinois at Urbana Champaign
"""

import argparse

def get_args():
    """! Command line parser for experiments"""
    parser = argparse.ArgumentParser(description='Deep Clustering for '
                                                 'Audio Source '
                                                 'Separation '
                                                 'Experiment')
    parser.add_argument("--train", type=str,
                        help="Path for the training dataset",
                        default=None)
    parser.add_argument("--test", type=str,
                        help="Path for the testing dataset",
                        default=None)
    parser.add_argument("--val", type=str,
                        help="Path for the validation dataset",
                        default=None)
    parser.add_argument("--n_train", type=int,
                        help="""Reduce the number of training 
                            samples to this number.""", default=None)
    parser.add_argument("--n_test", type=int,
                        help="""Reduce the number of testing 
                            samples to this number.""", default=None)
    parser.add_argument("--n_val", type=int,
                        help="""Reduce the number of evaluation 
                            samples to this number.""", default=None)
    parser.add_argument("-nl", "--n_layers", type=int,
                        help="""The number of layers of the BLSTM 
                        encoder""", default=2)
    parser.add_argument("-ed", "--embedding_depth", type=int,
                        help="""The depth of the embedding""",
                        default=10)
    parser.add_argument("-hs", "--hidden_size", type=int,
                        help="""The size of the LSTM cells """,
                        default=10)
    parser.add_argument("-bs", "--batch_size", type=int,
                        help="""The number of samples in each batch. 
                        Warning: Cannot be less than the number of 
                        the validation samples""", default=64)
    parser.add_argument("-name", "--experiment_name", type=str,
                        help="""The name or identifier of this 
                        experiment""",
                        default='A sample experiment'),
    parser.add_argument("-l", "--labels_mask", type=str,
                        help="""The type of masks that you want to 
                        use -- 'ground_truth' or 'duet'""",
                        default='duet', choices=['duet',
                                                 'ground_truth'])
    parser.add_argument("-cad", "--cuda_available_devices", type=int,
                        nargs="+",
                        help="""A list of Cuda IDs that would be 
                        available for runnign this experiment""",
                        default=[0])
    parser.add_argument("--num_workers", type=int,
                        help="""The number of cpu workers for 
                        loading the data, etc.""", default=3)
    parser.add_argument("--epochs", type=int,
                        help="""The number of epochs that the 
                        experiment should run""", default=50)
    parser.add_argument("--eval_per", type=int,
                        help="""The number of training epochs in 
                        order to run an evaluation""", default=5)
    parser.add_argument("-lr", "--learning_rate", type=float,
                        help="""Initial Learning rate""", default=1e-3)
    parser.add_argument("--bidirectional", action='store_true',
                        help="""Bidirectional or not""")
    parser.add_argument("--save_best", type=int,
                        help="""The number of best models dependent 
                        on the metric you want to use that are going 
                        to be saved under the preferred logging model 
                        directory.""",
                        default=10)

    return parser.parse_args()