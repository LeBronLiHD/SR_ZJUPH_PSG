# coding: utf-8

import matplotlib.pyplot as plt
import os
import mne
import numpy as np
import params
import seaborn as sns
import pandas as pd
import utils
from sklearn.model_selection import train_test_split


# @static
def _split_train_test(ori_x, ori_y):
    """
    ori_x -> 1186, 23, 37026
    ori_y -> 1186
    """
    index = np.arange(len(ori_y))
    np.random.shuffle(index)
    ori_x = ori_x[index, :, :]
    ori_y = ori_y[index]
    cur_x_train, cur_y_train, cur_x_test, cur_y_test = train_test_split(ori_x, ori_y, test_size=(1 - params.SPLIT_RATIO), random_state=42)
    return cur_x_train, cur_y_train, cur_x_test, cur_y_test

# @static 
def _get_normal_edf(edf_data):
    new_data = []
    for j in range(params.DIMENSION):
        print("Normalizing ... Current Dimension ->", j)
        new_data.append(utils._normalization(edf_data[j][params.VALUE_IDX][0]))
        # new_data.append(edf_data[j][params.VALUE_IDX][0])
    print("Normalization Done!")
    return np.array(new_data)

# @static
def _generate_train_test(edf_data, txt_data):
    ori_x, ori_y = [], []
    for i in range(len(txt_data)):
        print("Generating Train&Test ... Current Index ->", i) if i % 50 == 0 else None
        ori_y.append(params.LABEL_DICT[txt_data[i]])
        ori_x_unit = []
        for j in range(params.DIMENSION):
            ori_x_unit.append(np.array(edf_data[j][round(params.UNIT_LENGTH_F * j):round(params.UNIT_LENGTH_F * j) + params.UNIT_LENGTH]).T)
        ori_x.append(np.array(ori_x_unit).T)
    return np.array(ori_x), np.array(ori_y)

# @static
def _get_train_test(patient):
    edf_data = utils.get_edf(path=patient)
    txt_data = open(utils.get_txt(path=patient), mode="r", encoding="UTF8").read().split("\n")
    txt_data = txt_data[:-1]
    print(len(txt_data), len(edf_data))
    ori_x, ori_y = _generate_train_test(_get_normal_edf(edf_data), txt_data)
    return _split_train_test(ori_x, ori_y)

def show_shape(x_train, y_train, x_test, y_test):
    x_train, y_train, x_test, y_test = np.array(x_train), np.array(y_train), np.array(x_test), np.array(y_test)
    print("x_train ->", x_train.shape, x_train[0].shape)
    print("y_train ->", y_train.shape)
    print("x_test  ->", x_test.shape, x_test[0].shape)
    print("y_test  ->", y_test.shape)

# @static
def _small_sample(data):
    index = np.arange(len(data))
    np.random.shuffle(index)
    index = index[0:round(len(index) * params.SAMPLE_RATE)]
    print("small-sample process done!")
    return data[index]

def get_train_test_v2():
    data = np.load(os.path.join(params.DATA_PATH, params.NPZ_DATA))
    X, y = data['x'], data['y']
    x_train, y_train, x_test, y_test = _split_train_test(X, y)
    show_shape(x_train, y_train, x_test, y_test)
    if params.SMALL_SAMPLE:
        x_train, y_train, x_test, y_test = _small_sample(x_train), _small_sample(y_train), _small_sample(x_test), _small_sample(y_test)
        show_shape(x_train, y_train, x_test, y_test)
    return x_train, y_train, x_test, y_test

def get_train_test():
    x_train, y_train, x_test, y_test = [], [], [], []
    for patient in params.PATIENTS:
        cur_x_train, cur_x_test, cur_y_train, cur_y_test = _get_train_test(patient=patient)
        x_train.extend(cur_x_train)
        y_train.extend(cur_y_train)
        x_test.extend(cur_x_test)
        y_test.extend(cur_y_test)
    x_train, y_train, x_test, y_test = np.array(x_train), np.array(y_train), np.array(x_test), np.array(y_test)
    show_shape(x_train, y_train, x_test, y_test)
    if params.SMALL_SAMPLE:
        x_train, y_train, x_test, y_test = _small_sample(x_train), _small_sample(y_train), _small_sample(x_test), _small_sample(y_test)
        show_shape(x_train, y_train, x_test, y_test)
    return x_train, y_train, x_test, y_test

# @static
def _generate_sequence(data):
    seq = params.SEQUENCE
    ret = []
    for i in range(len(data) - seq):
        ret.append(data[i:i+seq])
    return np.array(ret)

def generate_sequence(x_train, y_train, x_test, y_test):
    return _generate_sequence(x_train), _generate_sequence(y_train), _generate_sequence(x_test), _generate_sequence(y_test)
