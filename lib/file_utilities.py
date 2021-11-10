
import os
from . import htk
import numpy as np
import re
from collections import namedtuple
import datetime

# See parse_files for interpretation
RecordingInfoType = namedtuple('RecordingInfo',
                               ('site', 'label', 'start', 'features'))

def parse_files(filenames):
    """
    Given a set of filenames from high-frequency acoustic recording packages
    (HARPs), extract information about the location and start time of each file.
    :param filenames: list of HARP files
    :return: List of LocationTimeType, named tuple with
             .site - name of recording location
             .label - class name
             .start - start time of data)
             .features - feature data e.g. cepstral features
    """

    # Regular expression to parse the time and location from the filename
    # Learning about regular expressions is really useful for many types of
    # data processing.  For details on Python, see:
    #      https://docs.python.org/3/library/re.html
    # This regular expression uses named patterns that allow us to refer
    # to individual portions of the string that are matched.
    #     Syntax:  (?P<name>subpattern)
    # The verbose flag allows us to split this over multiple lines,
    # ignoring most whitespace.
    species_re = re.compile( """
        .*[\\\\/](?P<species>Lo|Gg)[AB]?[\\\\/].*   # /LoA/ /Gg/ etc. (or \Gg\)
    """, re.VERBOSE)

    loctime_re = re.compile("""
       .*CAL   # Expect some form of SOCAL recording region
       (?P<deployment>\d+)  # instrument deployment number
       (?P<site>[A-Za-z0-9]+)_[^_]*_  # instrument site name
       (?P<year>\d\d)(?P<month>\d\d)(?P<day>\d\d)_  # start date
       (?P<h>\d\d)(?P<min>\d\d)(?P<s>\d\d).*     # start time
    """, re.VERBOSE)

    metadata = []   # List of file metadata to be returned

    progress_everyN = 25
    print("Extracting information about files and loading features for ",
          f"{len(filenames)} recordings.")
    for idx, f in enumerate(filenames):
        if idx % progress_everyN == 0:
            print(f"Reading file {idx}/{len(filenames)}")
        # Extract species from filepath using a regular expression
        m = species_re.match(f)
        if m is None:
            raise ValueError('Unable to extract species name')
        species = m.group('species')
        # Use regular expression to parse filename for where this was
        # recorded and when
        m = loctime_re.search(os.path.basename(f))
        if m is None:
            raise ValueError(f"Unable to parse location and time from file {f}")
        # Extract date and time
        yr = int(m.group('year'))
        century = 100
        if yr < century:
            # Two digit year, put in current century
            current_century = datetime.datetime.now().year // century * century
            yr += current_century
        timestamp = datetime.datetime(
            year=yr, month=int(m.group('month')),
            day=int(m.group('day')), hour=int(m.group('h')),
            minute=int(m.group('min')), second=int(m.group('s'))
        )

        features = get_features(f)  # Retrieve echolocation clicks features

        metadata.append(
            RecordingInfoType(m.group('site'), species,
                              timestamp, features))

    return metadata


def get_files(dir_path, ext=".czcc", stop_after=np.inf):
    """
    get_files - Retrieve all files from specified directory
    that have the given extension.
    :param dir_path:  directory name
    :param ext:  file extension
    :param start_after:  Stops reading after stop_after files are found
       (Useful for debugging so that you don't have to read every file in
        when you are trying to get things working)
    :return:  List of files matching extension
    """

    file_list = []
    idx = 0
    for path, directory, files in os.walk(dir_path):
        # Only keep files that meet criteria
        for f in files:
            # Get file extension
            basename, file_ext = os.path.splitext(f)
            # UNIX files starting w/. are hidden
            hidden_file = basename.startswith('."')
            if file_ext.lower() == ext and not hidden_file:
                # Meet criteria to keep
                file_list.append(os.path.join(path, f))
                idx += 1
                if idx >= stop_after:
                    break  # truncate search

    return file_list

def get_features(filename):
    """
    Given a complete path to a file, load the features from an HTK
    format file.  (Cambridge University HTK:  https://htk.eng.cam.ac.uk/)
    :param filename:  Name of HTK feature file
    :return: numpy tensor of features:  Example X Features
    """

    # Read in the data and convert it to a numpy array
    htkfile = htk.HTKFile()
    htkfile.load(filename)
    data = np.array(htkfile.data)

    return data
