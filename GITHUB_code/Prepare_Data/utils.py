# -*- coding: utf-8 -*-
# @Time    : 12/22/2018 11:10 PM
# @Author  : Littletim
# @File    : utils.py
# @Software: PyCharm
# @mail    : taocheng01@gmail.com

import h5py
import numpy as np
import random
import os
# to get better performance, shuffle training and testing images
def load_dataset():
    train_dataset = h5py.File('datasets/train.h5', "r")
    train_images = train_dataset["train_data"][:]
    train_labels = train_dataset["train_label"][:]
    c = list(zip(train_images, train_labels))
    random.shuffle(c)
    train_images, train_labels = zip(*c)
    train_images = np.array(train_images)  # your train set features
    train_labels = np.array(train_labels)  # your train set labels

    test_dataset = h5py.File('datasets/test.h5', "r")
    test_images = test_dataset["test_data"][:]
    test_labels = test_dataset["test_label"][:]
    c = list(zip(test_images, test_labels))
    random.shuffle(c)
    test_images, test_labels = zip(*c)
    test_images = np.array(test_images)  # your train set features
    test_labels = np.array(test_labels)  # your train set labels

    train_labels.reshape(len(train_labels), 1)
    test_labels.reshape(len(test_labels), 1)

    # print(train_images.shape,train_labels.shape)
    # print(test_images.shape,test_labels.shape)

    return train_images, train_labels, test_images, test_labels
