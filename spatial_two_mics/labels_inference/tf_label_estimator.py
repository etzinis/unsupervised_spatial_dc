"""!
@brief An estimator of TF masks depending on Blind Source Separation
Algorithms or even the energy in each bin (Ground Truth).

@author Efthymios Tzinis {etzinis2@illinois.edu}
@copyright University of Illinois at Urbana Champaign
"""

import numpy as np
import os
import sys
from pprint import pprint

root_dir = os.path.join(
    os.path.dirname(os.path.realpath(__file__)),
    '../../')
sys.path.insert(0, root_dir)
import spatial_two_mics.labels_inference.ground_truth as gt_inference
import spatial_two_mics.labels_inference.duet_mask_estimation as \
    duet_kmeans_inference


class TFMaskEstimator(object):
    """
    This is a general compatible class for encapsulating the label
    inference / a TF max for mixtures of signals coming from 2
    microphones.
    """
    def __init__(self,
                 inference_method=None):
        if inference_method.lower() == "ground_truth":
            self.label_inference = gt_inference
        elif inference_method.lower() == "duet_kmeans":
            self.label_inference = duet_kmeans_inference
        else:
            raise NotImplementedError("Inference Method: {} is not yet "
                  "implemented.".format(inference_method))

    def infer_mixture_labels(self,
                             mixture_info):
        """
        :param mixture_info:
        mixture_info = {
            'm1_raw': numpy array containing the raw m1 signal,
            'm2_raw': numpy array containing the raw m2 signal,
            'm1_tf': numpy array containing the m1 TF representation,
            'm2_tf': numpy array containing the m2 TF representation,
            'sources_raw': a list of numpy 1d vectors containing the
            sources ,
            'sources_tf': a list of numpy 2d vectors containing the
             TF represeantations of the sources ,
            'delayed_sources_raw': a list of numpy 1d vectors containing
            the sources delayed with some tau,
            'delayed_sources_tf': a list of numpy 2d vectors
            containing the TF representations of the delayed signals,
            'amplitudes': the weights that each source contributes to
            the mixture of the second microphone
        }

        :return: A TF representation with each TF bin to correspond
        to the source which the algorithm predicts that is dominating
        """

        return self.label_inference.infer_mask(mixture_info)


def example_of_usage():
    """!
    How the class of Audio mixtures should be called"""

    import os
    import sys
    root_dir = os.path.join(
        os.path.dirname(os.path.realpath(__file__)),
        '../../')
    sys.path.insert(0, root_dir)
    import spatial_two_mics.examples.mixture_example as me
    import spatial_two_mics.utils.audio_mixture_constructor as \
        mix_constructor

    mixture_info = me.mixture_info_example()
    mixture_creator = mix_constructor.AudioMixtureConstructor(
        n_fft=1024, win_len=400, hop_len=200, mixture_duration=2.0,
        force_delays=[-1, 1])

    tf_mixtures = mixture_creator.construct_mixture(mixture_info)

    duet_estimator = TFMaskEstimator(inference_method='duet_Kmeans')

    tf_labels = duet_estimator.infer_mixture_labels(tf_mixtures)
    print("DUET Kmeans")
    pprint(tf_labels.shape)

    ground_truth_estimator = TFMaskEstimator(
        inference_method='ground_truth')

    gt_labels = ground_truth_estimator.infer_mixture_labels(tf_mixtures)
    print("Ground Truth")
    pprint(gt_labels.shape)

    n_bins = np.product(gt_labels.shape)
    print("Estimation differs at {} out of {} points".format(
          min(np.count_nonzero(abs(gt_labels-tf_labels)),
              n_bins - np.count_nonzero(abs(gt_labels - tf_labels))),
          n_bins))


if __name__ == "__main__":
    example_of_usage()
