Description:
    Given a directory of HARP (High-frequency Acoustic Recording Packages) files for two dolphin species Gg and Lo;
    this program serves to train a sequential neural network to accurately distinguish between species.

Step by Step Process:

1. Extract HARP acoustic files from included "features" directory
    - parse files and store meta data in dictionary with key = date.time and value = [(.site, label, .start, .features), ...]
    - 1 dictionary per species type; either "Gg" or "Lo"
    - split "training" and "testing" days for both species

2. Organize "features" metadata for Training Days
    - Append .features to training data list
    - Append a corresponding label to labels list i.e. [1,0] for Gg and [0,1] for Lo
    - create weights for each species based on frequency of occurence in data

3. Organize "features" metadata for Testing Days
    - For every feature we will group its data into batches of 100 when possible
    - For every batch a single corresponding label will be created which will be used when creating confusion matrix
    - When testing the trained model our output will be based on 1 batch

4. Model Training
    - we use a Sequential neural network with ReLU activation layers to avoid disappearing gradient
    - we use a Softmax activation output layer to classify data as Gg or Lo based on the probability that it is one
    or the other

5. Model Testing
    - we test our model with our previously created batches and determine what classification it falls under

6. Confusion Matrix
    - display our True Positives, False Positives, True Negatives, False Negatives

Packages Required:
A Conda YML file is provided that will install the packages
needed for this project.  See conda.yml for instructions on how to
set up a tensorflow environment in Anaconda.

If you are not using Anaconda, you should be able to install these
packages manually with pip.

