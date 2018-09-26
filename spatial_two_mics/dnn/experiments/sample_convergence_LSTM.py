"""!
@brief A simple experiment on how LSTM converge

@author Efthymios Tzinis {etzinis2@illinois.edu}
@copyright University of illinois at Urbana Champaign
"""

import os
import sys
import torch
import time
import numpy as np
import copy
from pprint import pprint

root_dir = os.path.join(
    os.path.dirname(os.path.realpath(__file__)),
    '../../../')
sys.path.insert(0, root_dir)

import spatial_two_mics.dnn.models.simple_LSTM_encoder as LSTM_enc
import spatial_two_mics.dnn.losses.affinity_approximation as \
    affinity_losses
import spatial_two_mics.dnn.utils.dataset as data_generator
import spatial_two_mics.dnn.utils.data_conversions as converters
import spatial_two_mics.dnn.utils.experiment_command_line_parser as \
    parser


def convergence_of_LSTM(args):
    print(args)


if __name__ == "__main__":
    args = parser.get_args()
    convergence_of_LSTM(args)