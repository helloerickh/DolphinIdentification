from collections import defaultdict


def split_by_day(recordings):
    """
    Given a list of RecordingInfoTypes, split them into a list of lists where
    each list represents one day of recording.
    :param recordings:
    :return:
    """

    bydate = defaultdict(list)
    for r in recordings:
        # For each recording, append to a list keyed on the start day
        date = r.start.date()
        bydate[date].append(r)

    return bydate