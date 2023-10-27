# Desc: Configuration file for the project

import os
CURRENT_DIR = os.path.dirname(os.path.realpath(__file__))

# Path to the dataset
DATASET_DIR = CURRENT_DIR + '/data'
IMG_DIR = CURRENT_DIR + '/data/images'
LOG_DIR = CURRENT_DIR + '/lightning_logs'

# Model parameters
BATCH_SIZE = 16
LEARNING_RATE = 0.001
MAX_EPOCHS = 1

# Reproducibility for splitting the dataset
SEED = 42

# training validation split
# 70-10-20
TRAIN_RATIO = 0.7
VAL_RATIO = 0.1
TEST_RATIO = 0.2