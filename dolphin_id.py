# Might make your life easier for appending to lists
from collections import defaultdict

# Third party libraries
import numpy as np
# Only needed if you plot your confusion matrix
import matplotlib.pyplot as plt
import keras
from keras.models import Sequential
from keras.layers import Dense
from keras import regularizers
from sklearn.metrics import confusion_matrix
import seaborn

# our libraries

# Any other modules you create
from create_data import create_train_data, create_test_data
from pre_processing import pre_processing


def dolphin_classifier(data_directory):
    """
    Neural net classification of dolphin echolocation clicks to species
    :param data_directory:  root directory of data
    :return:  None
    """

    plt.ion()   # enable interactive plotting

    # use this to get all files
    use_onlyN = np.Inf

    gg_day_dict, gg_train_days, gg_test_days, lo_day_dict, lo_train_days, lo_test_days = pre_processing(data_directory, use_onlyN)

    X_train, Y_train, class_weights = create_train_data(gg_train_days, lo_train_days, gg_day_dict, lo_day_dict)

    X_test, Y_test = create_test_data(gg_test_days, lo_test_days, gg_day_dict, lo_day_dict)

    model = Sequential()
    model.add(Dense(100, input_dim=20, activation='relu', kernel_regularizer='l2'))
    model.add(Dense(100, activation='relu', kernel_regularizer='l2'))
    model.add(Dense(100, activation='relu', kernel_regularizer='l2'))
    model.add(Dense(100, activation='relu', kernel_regularizer='l2'))
    model.add(Dense(2, activation='softmax'))

    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

    nn_model = model.fit(X_train, Y_train, epochs=5, batch_size=16, class_weight=class_weights)

    pred_classes = []
    for x in X_test:
        out = model.predict(x)
        sum_prob = np.sum(np.log(out), axis=0)
        pred_classes.append(np.argmax(sum_prob))

    confoos = confusion_matrix(Y_test,pred_classes)
    error_rate = (confoos[0][1] + confoos[1][0]) / (confoos[0][0] + confoos[1][1])
    print("The error rate is: ", error_rate)

    seaborn.set_theme(color_codes=True)
    plt.figure(1, figsize=(9, 6))
    plt.title("Confusion Matrix")
    seaborn.set_theme(font_scale=1.4)
    ax = seaborn.heatmap(confoos, annot=True, cmap="YlGnBu", cbar_kws={'label': 'Scale'}, fmt='g')

    labels = ["Gg", "Lo"]
    ax.set_xticklabels(labels)
    ax.set_yticklabels(labels)
    ax.set(ylabel="True Label", xlabel="Predicted Label")

    plt.savefig("confusion_matrix", bbox_inches='tight', dpi=300)
    plt.title("Dolphin Classification Confusion Matrix")
    plt.show()


    #raise NotImplementedError


if __name__ == "__main__":
    dolphin_classifier(["./features/Gg", "./features/Lo/A"])