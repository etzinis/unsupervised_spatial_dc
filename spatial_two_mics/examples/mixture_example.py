"""!
@brief A simple example of how a compact mixture should look like

@author Efthymios Tzinis {etzinis2@illinois.edu}
@copyright University of Illinois at Urbana Champaign
"""

import os
import sys
root_dir = os.path.join(
    os.path.dirname(os.path.realpath(__file__)),
    '../../')
sys.path.insert(0, root_dir)

from spatial_two_mics.config import TIMIT_PATH
import numpy as np


def mixture_info_example():
    ex = {'positions':
                  {'amplitudes': np.array([0.73382382,
                                                0.26617618]),
                   'd_thetas': np.array([1.06829948]),
                   'distances': {'m1m1': 0.0,
                                 'm1m2': 0.03,
                                 'm1s1': 3.015,
                                 'm1s2': 3.0072529608785676,
                                 'm2m1': 0.03,
                                 'm2m2': 0.0,
                                 'm2s1': 2.985,
                                 'm2s2': 2.9928046426867034,
                                 's1m1': 3.015,
                                 's1m2': 2.985,
                                 's1s1': 0.0,
                                 's1s2': 3.054656422155759,
                                 's2m1': 3.0072529608785676,
                                 's2m2': 2.9928046426867034,
                                 's2s1': 3.054656422155759,
                                 's2s2': 0.0},
                   'taus': np.array([1.39941691, 0.67397403]),
                   'thetas': np.array([0., 1.06829948]),
                   'xy_positons': np.array([[3., 0.],
                                         [1.44484569, 2.62914833]])},
     'sources_ids': [{'gender': 'f',
                     'sentence_id': 'sa1',
                     'speaker_id': 'flbw0',
                     'wav_path': os.path.join(TIMIT_PATH,
                                 'test/dr4/flbw0/sa1.wav')},
                    {'gender': 'm',
                     'sentence_id': 'sa2',
                     'speaker_id': 'mbns0',
                     'wav_path': os.path.join(TIMIT_PATH,
                                 'test/dr4/mbns0/sa2.wav')}
                     ]}

    return ex
