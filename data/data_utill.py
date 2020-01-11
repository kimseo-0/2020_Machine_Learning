import h5py
import definition
import numpy as np
import os

def one_hot(m, y, num_of_class) :
    one_hot_matrix = np.zeros((m,num_of_class))
    for i in range(m) :
        one_hot_matrix[i][y[i]] = 1 # 0-5 데이터

    return one_hot_matrix

def load_sign_dataset() :
    train_dataset = h5py.File(os.path.join(definition.ROOT_DIR,'data\train_signs.h5'), 'r')
    x_train = np.array(train_dataset['train_set_x'][:])
    y_train = np.array(train_dataset['train_set_y'][:])
    (m_train, height_train, width_train, channel_train) = x_train.shape
    y_train = y_train.reshape(m_train, -1)

    test_dataset = h5py.File(os.path.join(definition.ROOT_DIR,'data\test_signs.h5'), 'r')
    x_test = np.array(test_dataset['test_set_x'][:])
    y_test = np.array(test_dataset['test_set_y'][:])
    (m_test, height_test, width_test, channel_test) = x_test.shape
    y_test = y_test.reshape(m_test, -1)

    classes = np.array(train_dataset['list_classes'][:])
    num_of_class = classes.shape[0]

    one_hot_y_train = one_hot(m_train, y_train, num_of_class)
    one_hot_y_test = one_hot(m_test, y_test, num_of_class)

    dim = height_train * width_train * channel_train

    return x_train, one_hot_y_train, x_test, one_hot_y_test, dim

def flatten_and_reshape(x, y) :
    (m, _, _, _) = x.shape
    x_reshape = x.reshape(m, -1).T
    y_reshape = y.reshape(m, -1).T

    return x_reshape, y_reshape

def cetralize(x) :
    x_centralize = x / 255.

    return x_centralize