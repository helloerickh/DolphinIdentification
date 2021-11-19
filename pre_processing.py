import lib.file_utilities as util
from lib.partition import split_by_day
from sklearn.model_selection import train_test_split
import os

def pre_processing(data_directory, use_onlyN):
    gg_dir = os.path.abspath(data_directory[0])
    lo_dir = os.path.abspath(data_directory[1])

    # get list of click files for each species
    ggfiles = util.get_files(gg_dir, ".czcc", use_onlyN)
    lofiles = util.get_files(lo_dir, ".czcc", use_onlyN)

    # create lists of tuples (.site, .label, .start, .features) for each species
    ggmeta_data = util.parse_files(ggfiles)
    lometa_data = util.parse_files(lofiles)
    test_meta_data = ggmeta_data[0]

    # create dictionaries keyed by day
    # key=datetime.start value=list[tuples (.site, label, .start, .features)]
    gg_day_dict = split_by_day(ggmeta_data)
    lo_day_dict = split_by_day(lometa_data)

    # create lists of days in dictionaries
    gg_keys = list(gg_day_dict.keys())
    lo_keys = list(lo_day_dict.keys())

    gg_train_test_days = train_test_split(gg_keys, test_size=0.33, random_state=42)
    lo_train_test_days = train_test_split(lo_keys, test_size=0.33, random_state=42)

    gg_train_days = gg_train_test_days[0]
    gg_test_days = gg_train_test_days[1]

    lo_train_days = lo_train_test_days[0]
    lo_test_days = lo_train_test_days[1]

    return gg_day_dict, gg_train_days, gg_test_days, lo_day_dict, lo_train_days, lo_test_days