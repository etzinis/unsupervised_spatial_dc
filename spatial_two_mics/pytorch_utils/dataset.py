"""!
@brief A dataset creation which is compatible with pytorch framework

@author Efthymios Tzinis {etzinis2@illinois.edu}
@copyright University of illinois at Urbana Champaign
"""

import os
import sys

root_dir = os.path.join(
           os.path.dirname(os.path.realpath(__file__)),
           '../../')
sys.path.insert(0, root_dir)

import spatial_two_mics.