import numpy as np


def initialize_parameter(layer_dim):
    params = {}

    for i in range(1, len(layer_dim)):
        params['w' + str(i)] = np.random.randn(layer_dim[i - 1], layer_dim[i]) * 0.01
        params['b' + str(i)] = 0

    return params


def linear_forward(w, b, a):
    z = np.matmul(w.T, a) + b
    linear_cache = w, b, a

    return z, linear_cache


def linear_backward(dz, linear_cache):
    w, b, a = linear_cache
    m = a.shape[1]

    dw = np.matmul(a, dz.T)
    db = np.sum(dz)
    da = np.matmul(w, dz)

    return dw, db, da


def sigmoid(z):
    a = 1 / (1 + np.exp(-z))
    activation_cache = z

    return a, activation_cache


def relu(z):
    a = np.maximum((0, z))
    activation_cache = z

    return a, activation_cache


def relu_backward(da, activation_cache):
    z = activation_cache

    da_dz = np.ones(z.shape)
    da_dz[z < 0] = 0

    dz = da_dz * da

    return dz


def leaky_relu(z):
    a = np.maximum(0.01 * z, z)
    activation_cache = z

    return a, activation_cache


def leaky_relu_backward(da, activation_cache):
    z = activation_cache

    da_dz = np.ones(z.shape)
    da_dz[z < 0] = 0.01

    dz = da_dz * da

    return dz


def softmax(z):
    activation_cache = z
    z = z - np.max(z, axis=0, keepdims=True)
    a = np.exp(z) / np.sum(np.exp(z), axis=0, keepdims=True)

    return a, activation_cache


def softmax_backward(da, activation_cache):
    z = activation_cache
    a = np.exp(z) / np.sum(np.exp(z), axis=0, keepdims=True)

    (r, c) = da.shape
    da_dz = np.zeros((r, r))
    for i in range(r):
        for j in range(r):
            if i == j:
                da_dz[i][j] = a[i] * (1 - a[i])
            else:
                da_dz[i][j] = - a[i] * a[j]

    dz = np.matmul(da_dz, da)

    return dz


def linear_activation_forward(w, b, a, activation):
    z, linear_cache = linear_forward(w, b, a)

    if activation == 'sigmoid':
        a, activation_cache = sigmoid(z)
    elif activation == 'leaky_relu':
        a, activation_cache = leaky_relu(z)
    elif activation == 'relu':
        a, activation_cache = relu(z)
    elif activation == 'softmax':
        a, activation_cache = softmax(z)

    linear_activation_cache = (linear_cache, activation_cache)

    return a, linear_activation_cache


def linear_activation_backward(da, linear_activation_cache, activation):
    (linear_cache, activation_cache) = linear_activation_cache

    dz = None
    if activation == 'relu':
        dz = relu_backward(da, activation_cache)
    elif activation == 'leaky_relu':
        dz = leaky_relu_backward(da, activation_cache)
    elif activation == 'softmax':
        dz = softmax_backward(da, activation_cache)
    dw, db, da = linear_backward(dz, linear_cache)

    return dw, db, da


def forward(x, params, num_of_layers):
    forward_cache = []

    a = x
    for i in range(1, num_of_layers - 1):
        a, linear_activation_cache = linear_activation_forward(params['w' + str(i)], params['b' + str(i)], a, 'relu')
        forward_cache.append(linear_activation_cache)

    a, linear_activation_cache = linear_activation_forward(params['w' + str(num_of_layers - 1)],
                                                           params['b' + str(num_of_layers - 1)], a, 'softmax')
    forward_cache.append(linear_activation_cache)

    return a, forward_cache


def backward(y, a, num_of_layers, forward_cache, last_activation):
    grads = {}
    da = - y / a  # cross entrophy 일때

    dw, db, da = linear_activation_backward(da, forward_cache[-1], 'softmax')
    grads['dw' + str(num_of_layers - 1)] = dw
    grads['db' + str(num_of_layers - 1)] = db
    grads['da' + str(num_of_layers - 1)] = da
    for i in reversed(range(1, num_of_layers - 1)):
        dw, db, da = linear_activation_backward(da, forward_cache[i - 1], 'relu')
        grads['dw' + str(i)] = dw
        grads['db' + str(i)] = db
        grads['da' + str(i)] = da

    return grads


def update_params(params, grads, learning_rate, num_of_layers, lam, m):
    for i in range(1, num_of_layers):
        params['w' + str(i)] = params['w' + str(i)] * (1 - learning_rate * lam / m) - grads[
            'dw' + str(i)] * learning_rate
        params['b' + str(i)] = grads['db' + str(i)] * learning_rate

    return params


def compute_cost(y, a, params, lam, num_of_layers):
    m = y.shape[1]
    cost = np.sum(-(y * np.log(a))) / m

    regularlize_term = 0
    for i in range(1, num_of_layers - 1):
        w = params['w' + str(i)]
        regularlize_term = regularlize_term + np.sum(np.square(w))

    regularlize_term = regularlize_term * lam / (2 * m)

    cost = cost + regularlize_term

    return cost
