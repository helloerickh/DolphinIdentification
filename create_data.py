import numpy as np


def create_train_data(train_days_gg, train_days_lo, gg_day_dict, lo_day_dict):
    train_tensor_examples = []
    train_tensor_labels = []

    # iterate through gg training days
    for day in train_days_gg:
        # print("Day: {}\n".format(day))
        # iterate through recordings in day
        for recording in gg_day_dict[day]:
            # get row, col to make correct amount of labels
            row, col = recording.features.shape
            # print("This recording has rows: {} cols: {} \n".format(row, col))
            train_tensor_examples.append(recording.features)
            # create row amount of gg labels
            train_tensor_labels.append([[1, 0]] * row)

    # iterate through lo training days
    for day in train_days_lo:
        # print("Day: {}\n".format(day))
        # iterate through recordings in day
        for recording in lo_day_dict[day]:
            # get row, col to make correct amount of labels
            row, col = recording.features.shape
            # print("This recording has rows: {} cols: {}\n".format(row, col))
            train_tensor_examples.append(recording.features)
            # create row amount of lo labels
            train_tensor_labels.append([[0, 1]] * row)

    # stack all recordings on top of one another, creates (total # training examples, 20) ndarray
    big_train_examples = np.concatenate(train_tensor_examples, axis=0)
    # combines all labels together, create vector of size (total # training examples)
    big_train_labels = np.concatenate(train_tensor_labels, axis=0)

    # print("Shape of training data: {}\n".format(big_train_examples.shape))
    # print("Size of label vector: {}\n".format(big_train_labels.size))

    return big_train_examples, big_train_labels


# similar to making training data, but we don't combine everything
# instead we create a list that we can iterate through and test on each element separately
def create_test_data(test_days_gg, test_days_lo, gg_day_dict, lo_day_dict):
    test_tensor_examples = []
    test_tensor_labels = []

    # iterate through gg test days
    for day in test_days_gg:
        # iterate through recording sessions in a day
        for recording in gg_day_dict[day]:
            # if recording session has less than 100 recordings, skip
            if len(recording.features) < 100:
                continue
            # get as many 100 batches as possible from recording session
            groups, groups_labels = hunnit_group(recording)
            test_tensor_examples.append(groups)
            test_tensor_labels.append(groups_labels)

    # iterate through lo test days
    for day in test_days_lo:
        # iterate through recording sessions in a day
        for recording in lo_day_dict[day]:
            # if recording session has less than 100 recordings, skip
            if len(recording.features) < 100:
                continue
            # get as many 100 batches as possible from recording session
            groups, groups_labels = hunnit_group(recording)
            test_tensor_examples.append(groups)
            test_tensor_labels.append(groups_labels)

    # we can iterate through these lists to test on each batch separately
    big_test_examples = np.concatenate(test_tensor_examples, axis=0)
    big_test_labels = np.concatenate(test_tensor_labels, axis=0)

    return big_test_examples, big_test_labels

# helper function to create batches, takes a metadata tuple
def hunnit_group(recording):
    recording_tensor = recording.features
    print("Number of rows in features {}".format(len(recording_tensor)))
    if recording.label == "Gg":
        print("Label is Gg: [1,0]\n")
        label = [1,0]
    else:
        print("Label is Lo: [0,1]\n")
        label = [0,1]
    # list comprehension to grab as many 100 clicks as we can
    hunnit_batches = [recording_tensor[x:x+100] for x in range(0, len(recording_tensor), 100) if ((len(recording_tensor) - x) >= 100)]
    # for each 100 clicks batch create a corresponding label vector
    label_batches = [np.array([label] * 100)] * len(hunnit_batches)
    return hunnit_batches, label_batches

