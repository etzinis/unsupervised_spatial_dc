"""!
@brief Dataloader for timit dataset in order to store in an internal
python dictionary structure the whole timit dataset.

@author Efthymios Tzinis {etzinis2@illinois.edu}
@copyright University of illinois at Urbana Champaign
"""

import os
import sys

root_dir = os.path.join(
           os.path.dirname(os.path.realpath(__file__)),
           '../../')
sys.path.insert(0, root_dir)


if __name__ == "__main__":
    print("yoler")
