# coding: utf-8

import os


DATA_PATH = ".\\data"

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

UNIT_LENGTH_F = 36441088/NUM_LABEL
UNIT_LENGTH = round(36441088/NUM_LABEL)
INPUT_SHAPE = (UNIT_LENGTH, DIMENSION)
INPUT_SHAPE_2 = (None, UNIT_LENGTH, DIMENSION)
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

SPLIT_RATIO = 0.8
BATCH_SIZE = 32
EPOCH_NUM = 100
VERBOSE = 1
LEARNING_RATE = 0.001

SMALL_SAMPLE = True
SAMPLE_RATE = 0.1
