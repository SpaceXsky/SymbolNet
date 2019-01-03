# -*- coding: utf-8 -*-
# @Time    : 12/22/2018 11:00 PM
# @Author  : Littletim
# @File    : prepare_data.py
# @Software: PyCharm
# @mail    : taocheng01@gmail.com
import cv2
import os
import h5py
import sys
from multiprocessing import Pool
import multiprocessing


##preprocessing image and put them into h5 file
# using multiprocessing
def COLOR2GREY(src_img):
    return cv2.cvtColor(src_img, cv2.COLOR_BGR2GRAY)


def GaussianBluring(src_img, kernel):
    return cv2.GaussianBlur(src_img, kernel, 0)


def read_preprocess(path):
    img = cv2.imread(path)
    img = COLOR2GREY(img)
    img = GaussianBluring(img, (3, 3))
    img = (255 - img) / 255
    return img


def main():
    train_dataset = h5py.File('train.h5', "w")
    test_dataset = h5py.File('test.h5', "w")

    train_data = []
    train_label = []
    test_data = []
    test_label = []

    categories = os.listdir(r'C:\Users\89228\Desktop\Data Science\extracted_images')
    cores = multiprocessing.cpu_count()
    pool = Pool(cores)
    for i in range(len(categories)):
        root_path = r'C:\Users\89228\Desktop\Data Science\extracted_images' + '\\' + categories[i]
        paths = os.listdir(root_path)[:1000]

        label = []
        for j in range(len(paths)):
            paths[j] = root_path + '\\' + paths[j]
            label += [i]

        data = pool.map(read_preprocess, paths)
        num = len(paths)

        test_data += data[0:int(num / 5)]
        test_label += label[0:int(num / 5)]

        train_data += data[int(num / 5):]
        train_label += label[int(num / 5):]

    train_dataset['train_data'] = train_data
    train_dataset['train_label'] = train_label

    test_dataset['test_data'] = test_data
    test_dataset['test_label'] = test_label

    print(len(train_data), len(train_label))
    print(len(test_data), len(test_label))
    train_dataset.close()
    test_dataset.close()


if __name__ == '__main__':
    sys.exit(main())
