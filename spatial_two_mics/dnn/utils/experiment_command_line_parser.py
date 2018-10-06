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
    parser.add_argument("--dataset", type=str,
                        help="Dataset name",
                        default="timit")
    parser.add_argument("--n_sources", type=int,
                        help="How many sources in each mix",
                        default=2)
    parser.add_argument("--n_samples", type=int, nargs='+',
                        help="How many samples do u want to be "
                             "created for train test val",
                        default=[256, 64, 128])
    parser.add_argument("--genders", type=str, nargs='+',
                        help="Genders that will correspond to the "
                             "genders in the mixtures",
                        default=['m'])
    parser.add_argument("-f", "--force_delays", nargs='+', type=int,
                        help="""Whether you want to force integer 
                        delays of +- 1 in the sources e.g.""",
                        default=[-1, 1])
    parser.add_argument("-nl", "--n_layers", type=int,
                        help="""The number of layers of the LSTM 
                        encoder""", default=2)
    parser.add_argument("-ed", "--embedding_depth", type=int,
                        help="""The depth of the embedding""",
                        default=10)
    parser.add_argument("-hs", "--hidden_size", type=int,
                        help="""The size of the LSTM cells """,
                        default=10)
    parser.add_argument("-bs", "--batch_size", type=int,
                        help="""The number of samples in each batch""",
                        default=64)
    parser.add_argument("-name", "--experiment_name", type=str,
                        help="""The name or identifier of this 
                        experiment""",
                        default='A sample experiment'),
    parser.add_argument("-mt", "--labels_mask", type=str,
                        help="""The type of masks that you want to 
                        use -- 'ground_truth' or 'duet'""",
                        default='duet')
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
    parser.add_argument("--evaluate_per", type=int,
                        help="""The number of trianing epochs in 
                        order to run an evaluation""", default=5)
    parser.add_argument("--n_eval", type=int,
                        help="""Reduce the number of eavluation 
                        samples to this number.""", default=256)
    parser.add_argument("-lr", "--learning_rate", type=float,
                        help="""Initial Learning rate""", default=1e-3)
    parser.add_argument("--bidirectional", action='store_true',
                        help="""Bidirectional or not""")

    return parser.parse_args()