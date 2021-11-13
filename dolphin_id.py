# Might make your life easier for appending to lists
from collections import defaultdict

# Third party libraries
import numpy as np
# Only needed if you plot your confusion matrix
import matplotlib.pyplot as plt

# our libraries
from lib.partition import split_by_day
import lib.file_utilities as util

# Any other modules you create

import os


def dolphin_classifier(data_directory):
    """
    Neural net classification of dolphin echolocation clicks to species
    :param data_directory:  root directory of data
    :return:  None
    """

    plt.ion()   # enable interactive plotting


    #use_onlyN = np.Inf  # use this to get all files
    use_onlyN = 20



    #raise NotImplementedError

def pre_processing(dir, num_files):
    files = util.get_files(dir)
    meta_data = util.parse_files(files)



if __name__ == "__main__":
    # root directory of data, use os.path.abspath to avoid errors with diff operating systems
    data_directory = os.path.abspath("./features")
    dolphin_classifier(data_directory)