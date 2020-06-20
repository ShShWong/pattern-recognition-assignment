import numpy as np
import os
import random
import matplotlib.pyplot as plt
import h5py

"""
    generate a h5 data file, only run once
"""


def read_image(file_dir):
    """
    read all the subpath of the image and store in a list
    :param file_dir: the main path of the data
    :return: image list
    """

    label0 = []
    labelFor0 = []
    label1 = []
    labelFor1 = []
    for file in os.listdir(file_dir + '0/'):
        label0.append(file_dir + '0/' + file)
        labelFor0.append(0)
    for file in os.listdir(file_dir + '1/'):
        label1.append(file_dir + '1/' + file)
        labelFor1.append(1)

    image_list = np.hstack((label0, label1))
    label_list = np.hstack((labelFor0, labelFor1))

    # shuffle all the data
    temp = list(zip(image_list, label_list))
    random.shuffle(temp)

    image_list, label_list = zip(*temp)

    return image_list, label_list


def train_set(path='train/'):
    image_list, label_list = read_image(path)
    train_image = np.random.rand(len(image_list), 150, 150).astype('float32')
    train_label = np.random.rand(len(label_list)).astype('float32')
    for i in range(len(image_list)):
        train_image[i] = np.array(plt.imread(image_list[i]))
        train_label[i] = np.array(label_list[i])
    return train_image, train_label


def test_set(path='test/'):
    image_list, label_list = read_image(path)
    test_image = np.random.rand(len(image_list), 150, 150).astype('float32')
    test_label = np.random.rand(len(label_list)).astype('float32')
    for i in range(len(image_list)):
        test_image[i] = np.array(plt.imread(image_list[i]))
        test_label[i] = np.array(label_list[i])
    return test_image, test_label


train_image_orig, train_label_orig = train_set()
test_image_orig, test_label_orig = test_set()

f = h5py.File('./data.h5', 'w')
f.create_dataset('X_train', data=train_image_orig)
f.create_dataset('y_train', data=train_label_orig)
f.create_dataset('X_test', data=test_image_orig)
f.create_dataset('y_test', data=test_label_orig)
f.close()
