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


def dolphin_classifier(data_directory):
    """
    Neural net classification of dolphin echolocation clicks to species
    :param data_directory:  root directory of data
    :return:  None
    """

    plt.ion()   # enable interactive plotting

    use_onlyN = np.Inf  # debug, only read this many files for each species

    raise NotImplementedError


if __name__ == "__main__":
    data_directory = "path\to\data"  # root directory of data
    dolphin_classifier(data_directory)