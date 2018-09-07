"""!
@brief An estimator of TF masks depending on Blind Source Separation
Algorithms or even the energy in each bin (Ground Truth).

@author Efthymios Tzinis {etzinis2@illinois.edu}
@copyright University of Illinois at Urbana Champaign
"""

import os
import sys
from pprint import pprint

root_dir = os.path.join(
    os.path.dirname(os.path.realpath(__file__)),
    '../../')
sys.path.insert(0, root_dir)


class TFMaskEstimator(object):
    """
    This is a general compatible class for encapsulating the label
    inference / a TF max for mixtures of signals coming from 2
    microphones.
    """
    def __init__(self,
                 inference_method=None):
        if inference_method.lower() == "ground_truth":
            print("Build fucking ground truth first")
        else:
            raise NotImplementedError("Inference Method: {} is not yet "
                  "implemented.".format(inference_method))


def example_of_usage():
    """!
    How the class of Audio mixtures should be called"""

    ground_truth_estimator = TFMaskEstimator(
                             inference_method='Ground_truth')

if __name__ == "__main__":
    example_of_usage()
