"""!
@brief Infering the masking for eah tf bin based on DUET features,
mainly phase difference and after that a robust K-means estimation

@author Efthymios Tzinis {etzinis2@illinois.edu}
@copyright University of Illinois at Urbana Champaign
"""

import numpy as np
import sys
root_dir = '../../'
sys.path.insert(0, root_dir)
from spatial_two_mics.utils import robust_means_clustering as  \
     robust_kmeans


def infer_mask(mixture_info):
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
        'amplitudes': the weights that each source contributes to
        the mixture of the second microphone
    }

    :return: A tf 2d matrix corresponding to the dominating source
    for each TF bin [0,1,...,n_sources]
    """
    sources_complex_spectra = mixture_info['sources_tf']
    amplitudes = mixture_info['amplitudes']
    n_sources = len(sources_complex_spectra)

    assert len(amplitudes) == n_sources, "Length of weights: {} " \
                                         "should be equal to the " \
                                         "number of sources: {}" \
                                         "".format(len(amplitudes),
                                                   n_sources)

    same_dimensions = [(sources_complex_spectra[i].shape ==
                        sources_complex_spectra[0].shape)
                       for i in np.arange(len(sources_complex_spectra))]

    assert all(same_dimensions), "All arrays should have the same " \
                                 "dimensions. However, got sizes of {}"\
                                 "".format([x.shape for x in
                                            sources_complex_spectra])

    r = mixture_info['m1_tf'] / (mixture_info['m2_tf'] + 1e-7)
    phase_dif = np.angle(r) / np.linspace(1e-5, np.pi,
                              mixture_info['m1_tf'].shape[0])[:, None]

    d_feature = np.reshape(phase_dif, (np.product(phase_dif.shape), 1))
    r_kmeans = robust_kmeans.RobustKmeans(n_true_clusters=n_sources,
                                          n_used_clusters=n_sources+3)
    d_labels = r_kmeans.fit(d_feature, cut_outlier_in_norm=2.)
    d_feature_mask = np.reshape(d_labels, phase_dif.shape)

    zipped_tf_labels = d_feature_mask.astype(np.uint8)

    assert np.array_equal(d_feature_mask, zipped_tf_labels), \
        "Zipping the numpy matrix should not yield different labels"

    return zipped_tf_labels


