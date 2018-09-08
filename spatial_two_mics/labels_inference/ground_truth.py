"""!
@brief Infering the masking for eah tf bin independently based on the
maximum energy of the sources in each bin

@author Efthymios Tzinis {etzinis2@illinois.edu}
@copyright University of Illinois at Urbana Champaign
"""

import numpy as np
from pprint import pprint


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
        'delayed_sources_raw': a list of numpy 1d vectors containing
        the sources delayed with some tau,
        'delayed_sources_tf': a list of numpy 2d vectors
        containing the TF representations of the delayed signals,
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

    sources_complex_spectra = [amplitudes[i] * sources_complex_spectra[i]
                               for i in np.arange(n_sources)]

    tf_real_sources = [np.abs(tf_complex)
                       for tf_complex in sources_complex_spectra]

    mixture_tensor = np.dstack(tf_real_sources)
    dominating_source = np.argmax(mixture_tensor, axis=2)

    zipped_tf_labels = dominating_source.astype(np.uint8)

    assert np.array_equal(dominating_source, zipped_tf_labels), \
        "Zipping the numpy matrix should not yield different labels"

    return zipped_tf_labels


