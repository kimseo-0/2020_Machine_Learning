import numpy as np


def initialize_parameters(dim):  # dim 몇 차원인지
    parameters = {}

    parameters['w'] = np.random.randn(dim, 1) * 0.01
    parameters['b'] = 0

    return parameters


def linear_forward(w, b, a):
    z = np.matmul(w.T, a) + b
    linear_cache = w, b, a

    return z, linear_cache


def sigmoid(z):
    a = 1 / (1 + np.exp(-z))
    activation_cache = z

    return a, activation_cache


def leaky_relu(z):
    a = np.maximum(0.01 * z, z)
    activation_cache = z

    return a, activation_cache


def linear_activation_forward(w, b, a, activation):
    z, linear_cache = linear_forward(w, b, a)

    if activation == 'sigmoid':
        a, activation_cache = sigmoid(z)
    elif activation == 'leaky_relu':
        a, activation_cache = leaky_relu(z)

    linear_activation_cache = (linear_cache, activation_cache)

    return a, linear_activation_cache


def forward_and_backward(w, b, x, y):
    m = x.shape[1]

    a, activation_cache = sigmoid(np.matmul(w.T, x) + b)

    cost = (np.sum(-(y * np.log(a) + (1 - y) * np.log(1 - a)), axis=1) / (2 * m))
    cost = float(cost)

    dw = (np.matmul(x, (a - y).T)) / m  # dcost/dw
    db = np.sum((a - y)) / m  # dcost/db
    db = float(db)

    grads = {'dw': dw,
             'db': db}

    return grads, cost


def optimize(w, b, x, y, num_of_iterations, learning_rate):
    costs = []

    for i in range(num_of_iterations):

        grads, cost = forward_and_backward(w, b, x, y)

        w = w - learning_rate * grads['dw']
        b = b - learning_rate * grads['db']

        if i % 100 == 0:
            costs.append(cost)

    params = {'w': w,
              'b': b
              }

    return params, costs


def predict(params, x):
    m = x.shape(1)

    w = params['w']
    b = params['b']

    a, activation_cache = sigmoid(np.matmul(w.T, x) + b)

    prediction = np.zeros((1, m))
    for i in range(m):
        if a[0, i] > 0.8:
            prediction[0, i] = 1
        else:
            prediction[0, i] = 0

    return prediction
