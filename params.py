# coding: utf-8

import os


DATA_PATH = ".\\data"
NPZ_DATA = "data_sleep.npz"

def get_patients(path=DATA_PATH):
    patients = os.listdir(path)
    return [os.path.join(path, item) for item in patients]

PATIENTS = get_patients()
DIMENSION = 23
TIME_IDX = 1
VALUE_IDX = 0
TOL = 1e-12
EXPAND = 1e4
NUM_BLK = 1e4
NUM_LABEL = 1186
NUM_BLOCK = 36441088/NUM_BLK
NUM_BLOCK_INT = 600
NUM_BLk_SM = NUM_BLOCK/10
MIN_DIMEN = 2
SEQUENCE = 1

UNIT_LENGTH_F = 36441088/NUM_LABEL
UNIT_LENGTH = round(36441088/NUM_LABEL)
INPUT_SHAPE = (UNIT_LENGTH, DIMENSION)
INPUT_SHAPE_2 = (SEQUENCE, UNIT_LENGTH, DIMENSION)
RATIO = 6 # 2, 3, 6, 18, 54
INPUT_SHAPE_3 = (round(UNIT_LENGTH/RATIO), DIMENSION * RATIO, 1)

LABEL_DICT = {
    "W": 0,
    "N1": 1,
    "N2": 2,
    "N3": 3,
    "R": 4
}
NUM_CLASS = 5
CLASS = ["W", "N1", "N2", "N3", "R"]

SPLIT_RATIO = 0.8
BATCH_SIZE = 32
EPOCH_NUM = 10
VERBOSE = 1
LEARNING_RATE = 0.01
ADAM_BETA = [0.9, 0.999]

SMALL_SAMPLE = False
SAMPLE_RATE = 0.1

# configuration for tiny sleep
params_tiny = {
    "USE_RNN": True,
    "SAMPLE_RATE": 100.0,
    "IS_TRAINING": True,
    "INPUT_SIZE": UNIT_LENGTH,
    "DIMENSION": DIMENSION,
    "EPOCH_NUM": 10,
    "LEARNING_RATE": 1e-4,
    "ADAM_BETA_1": 0.9,
    "ADAM_BETA_2": 0.999,
    "RNN_LAYER": 1,
    "RNN_UNITS": 128,
    "L2_WEIGHT_DECAY": 1e-3
}

train_tiny = params_tiny.copy()
train_tiny.update({
    "SEQ_LENGTH": 1,
    "BATCH_SIZE": 16,
})

predict_tiny = params_tiny.copy()
predict_tiny.update({
    "SEQ_LENGTH": 1,
    "BATCH_SIZE": 1,
})

INPUT_SHAPE_4 = (train_tiny["SEQ_LENGTH"], 
                 params_tiny["INPUT_SIZE"], 
                 params_tiny["DIMENSION"])
