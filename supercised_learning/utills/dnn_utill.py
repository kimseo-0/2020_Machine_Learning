import numpy as np
import data.data_utill as data_utill


def initialize_parameter(layer_dim):
    params = {}

    for i in range(1, len(layer_dim)):
        params['w' + str(i)] = np.random.randn(layer_dim[i - 1], layer_dim[i]) * 0.01
        params['b' + str(i)] = np.zeros((layer_dim[i], 1))

    return params


def linear_forward(w, b, a):
    z = np.matmul(w.T, a) + b
    linear_cache = w, b, a

    return z, linear_cache


def linear_backward(dz, linear_cache):
    w, b, a = linear_cache
    m = a.shape[1]

    dw = np.matmul(a, dz.T) / m
    db = np.mean(dz, axis=1, keepdims=True)
    da = np.matmul(w, dz)

    return dw, db, da


def sigmoid(z):
    a = 1 / (1 + np.exp(-z))
    activation_cache = z

    return a, activation_cache


def relu(z):
    a = np.maximum(0, z)
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
    m = da.shape[1]
    z = activation_cache
    a = np.exp(z) / np.sum(np.exp(z), axis=0, keepdims=True)

    (r, c) = da.shape
    da_dz = np.zeros((r, r))
    for i in range(m):
        for j in range(r):
            for k in range(r):
                if j == k:
                    da_dz[j, k] = a[j, i] * (1 - a[j, i])
                else:
                    da_dz[j, k] = - a[j, i] * a[k, i]

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
        a, linear_activation_cache = linear_activation_forward(params['w' + str(i)], params['b' + str(i)], a,
                                                               'leaky_relu')
        forward_cache.append(linear_activation_cache)

    a, linear_activation_cache = linear_activation_forward(params['w' + str(num_of_layers - 1)],
                                                           params['b' + str(num_of_layers - 1)], a, 'softmax')
    forward_cache.append(linear_activation_cache)

    return a, forward_cache


def backward(y, a, num_of_layers, forward_cache, cost_function):
    grads = {}
    if cost_function == 'cross_entropy':
        da = - y / a  # cross entrophy 일때
    elif cost_function == 'min_square':
        da = a - y  # min square 일때

    dw, db, da = linear_activation_backward(da, forward_cache[-1], 'softmax')
    grads['dw' + str(num_of_layers - 1)] = dw
    grads['db' + str(num_of_layers - 1)] = db
    grads['da' + str(num_of_layers - 1)] = da
    for i in reversed(range(1, num_of_layers - 1)):
        dw, db, da = linear_activation_backward(da, forward_cache[i - 1], 'leaky_relu')
        grads['dw' + str(i)] = dw
        grads['db' + str(i)] = db
        grads['da' + str(i)] = da

    return grads


def gradient_clip(grad, limit):
    if np.linalg.norm(grad) > limit:
        grad *= (limit / np.linalg.norm(grad))

    return grad


def update_params(params, grads, learning_rate, num_of_layers, lam, m):
    for i in range(1, num_of_layers):
        params['w' + str(i)] = params['w' + str(i)] * (1 - learning_rate * lam / m) - gradient_clip(
            grads['dw' + str(i)], 1) * learning_rate
        params['b' + str(i)] = params['b' + str(i)] - learning_rate * gradient_clip(grads['db' + str(i)], 1)

    return params


def compute_cost(y, a, params, lam, num_of_layers):
    m = y.shape[1]
    cost = np.sum(-(y * np.log(a))) / m

    regularlize_term = 0
    for i in range(1, num_of_layers):
        w = params['w' + str(i)]
        regularlize_term = regularlize_term + np.sum(np.square(w))

    regularlize_term = regularlize_term * (lam / (2 * m))

    cost = cost + regularlize_term

    return cost


def optimize(params, x, y, num_of_layser, num_of_iteration, learning_rate, lam, batch_size, cost_function):
    mini_batches = data_utill.generate_random_mini_batches(x, y, batch_size)
    costs = []

    for mini_batch in mini_batches:
        mini_batch_x, mini_batch_y = mini_batch
        for i in range(num_of_iteration):
            a, forward_cache = forward(mini_batch_x, params, num_of_layser)
            cost = compute_cost(mini_batch_y, a, params, lam, num_of_layser)
            grads = backward(mini_batch_y, a, num_of_layser, forward_cache, cost_function)
            update_params(params, grads, learning_rate, num_of_layser, lam, batch_size)

            if i % 100 == 0:
                costs.append(cost)
                print(i, ' : ', cost)

    return params, costs


def predict(params, x, num_of_layers):
    a, _ = forward(x, params, num_of_layers)

    prediction = np.zeros(a.shape())
    for i in range(a.shape(0)):
        for j in range(a.shape(1)):
            if a[i, j] > 0.7:
                prediction[i, j] = 1

    return prediction
