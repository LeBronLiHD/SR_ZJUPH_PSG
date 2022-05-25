# coding: utf-8

# from crypt import static
from ast import Param
from distutils.log import set_threshold
from importlib.metadata import distribution
from nturl2path import pathname2url
import matplotlib.pyplot as plt
import os
import mne
import numpy as np
import params
import seaborn as sns
import pandas as pd


def get_txt(path):
    files = os.listdir(path)
    for file in files:
        if file[-4:] == ".txt":
            return os.path.join(path, file)
    print("No .txt file in ->", path)
    return None

# @static
def _load_raw_edf(path):
    raw_train = mne.io.read_raw_edf(path)
    return raw_train

def get_edf(path):
    """
    We Assume that Each Paitent has only One .edf File
    """
    files = os.listdir(path)
    for file in files:
        if file[-4:] == '.edf':
            print("Load .edf ->", os.path.join(path, file))
            return _load_raw_edf(os.path.join(path, file))
    print("No .edf file in ->", path)
    return None

# @static
def _normalization(data_ori):
    data = np.array([value * params.EXPAND for value in data_ori])
    if np.std(data) < params.TOL * params.EXPAND:
        print("Return data, np.std(data) is too Low! len data ->", len(data))
        return data
    return (data - np.mean(data))/np.std(data)/params.EXPAND

# @static
def _get_min_and_max(data):
    # if type(data) != 'list' or type(data) != 'numpy.ndarray':
    #     print("InValid Type of Min and Max!")
    #     return data, data
    return min(data), max(data)

# @static
def _count_range(data, thre_a, thre_b):
    if len(np.shape(data)) < params.MIN_DIMEN:
        print("InValid Shape! shape(data) ->", np.shape(data))
        return 1
    count = 0
    for item in data:
        if item >= thre_a and item < thre_b:
            count += 1
    return count

# @static
def _compression(data):
    compression = []
    for i in range(len(data)):
        if i == 0 or i % params.NUM_LABEL == 0:
            compression.append(data[i])
    return compression

# @static
def _get_distribution(data):
    data = data * params.EXPAND
    delta = (max(data) - min(data))/params.NUM_BLOCK
    threshold = [i * delta for i in range(round(params.NUM_BLOCK) + 1)]
    distribution = []
    for i in range(round(params.NUM_BLOCK)):
        distribution.append(_count_range(data, thre_a=threshold[i], thre_b=threshold[i + 1]))
    return distribution

# @static
def _plot_points(data_a, data_b, name):
    if len(data_a) != len(data_b):
        print("len(data_a) != len(data_b) Can not Polt! ( len a ->", len(data_a), "len b ->", len(data_b), ")")
        return None
    plt.figure(figsize=(20, 10))
    x = [i + 1 for i in range(len(data_a))]
    plt.plot(x, data_a, color='deeppink', marker='^', linewidth=0, markersize=42, alpha=0.7)
    plt.plot(x, data_b, color='green', marker='o', linewidth=0, markersize=42, alpha=0.7)
    for i in range(len(data_a)):
        point_a = [x[i], x[i]]
        point_b = [data_a[i], data_b[i]]
        plt.plot(point_a, point_b, color="deepskyblue")
        plt.plot([x[i]], [data_a[i]], color="deepskyblue", marker="s")
        plt.plot([x[i]], [data_b[i]], color="deepskyblue", marker="s")
    plt.title(name)
    plt.xlabel("Dimension")
    plt.ylabel("Value (Normalized)")
    plt.show()

# @static
def _plot_distribution(data, name):
    # distribution = _get_distribution(data)
    plt.figure(figsize=(20, 10))
    plt.hist(data, bins=params.NUM_BLOCK_INT, facecolor="deepskyblue", edgecolor="black", alpha=0.7)
    plt.xlabel("Dimension")
    plt.ylabel("Distribution")
    plt.title(name)
    plt.show()

# @static
def _plot_data(data, name):
    plt.figure(figsize=(20, 10))
    x = [i + 1 for i in range(len(data))]
    plt.plot(x, data, color='blue', marker='^', linewidth=1, markersize=4, alpha=0.5)
    plt.xlabel("Time")
    plt.ylabel("Value")
    plt.title(name)
    plt.show()

def compare_corr():
    dataset_edf = [get_edf(params.PATIENTS[i]) for i in range(len(params.PATIENTS))]
    
    for data in dataset_edf:
        new_data = []
        columns = []
        for i in range(params.DIMENSION):
            new_data.append(_compression(_normalization(data[i][params.VALUE_IDX][0])))
            columns.append("Dimension <" + str(i + 1) + ">")
            _plot_data(_compression(data[i][params.VALUE_IDX][0]), name="Dimension <" + str(i + 1) + ">")
        df = pd.DataFrame(data=np.array(new_data).T, columns=columns)
        plt.figure(figsize=(20, 20))
        sns.pairplot(df)
        plt.show()
        plt.figure(figsize=(20, 20))
        sns.heatmap(df.corr(), annot=True, cmap='RdBu_r')
        plt.show()

def compare_distribution():
    dataset_edf = [get_edf(params.PATIENTS[i]) for i in range(len(params.PATIENTS))]
    for data in dataset_edf:
        for i in range(params.DIMENSION):
            _plot_distribution(_normalization(data[i][params.VALUE_IDX][0]), name="Distribution of <" + str(i) + ">")

def compare_min_max():
    dataset_edf = [get_edf(params.PATIENTS[i]) for i in range(len(params.PATIENTS))]
    minset_set, maxset_set = [], []
    for data in dataset_edf:
        minset, maxset = [], []
        for i in range(params.DIMENSION):
            cur_min, cur_max = _get_min_and_max(_normalization(data[i][params.VALUE_IDX][0]))
            minset.append(cur_min)
            maxset.append(cur_max)
        minset_set.append(minset)
        maxset_set.append(maxset)
    if len(minset_set) == 2:
        """
        Currently We Only Have Two Patients
        """
        _plot_points(minset_set[0], minset_set[1], name="Min Value Comparsion of <" + str(0) + "-" + str(1) + ">")
        _plot_points(maxset_set[0], maxset_set[1], name="Max Value Comparsion of <" + str(0) + "-" + str(1) + ">")
        abs_min, abs_max = [np.abs(minset_set[0][i] - minset_set[1][i]) for i in range(len(minset_set[0]))], \
            [np.abs(maxset_set[0][i] - maxset_set[1][i]) for i in range(len(maxset_set[0]))]
        print("Comparsion in Min & Max Done!")
        return abs_min, abs_max
    else:
        raise NotImplementedError
