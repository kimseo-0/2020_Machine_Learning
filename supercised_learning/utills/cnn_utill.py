import numpy as np
import data.data_utill as data_utill
import supercised_learning.utils.dnn_utill as dnn_utill


def initialize_filter(filter_size, num_of_input_channel, num_of_output_channel):
    filter = np.random.randn(filter_size, filter_size, num_of_input_channel, num_of_output_channel) * 0.01

    return filter


def zero_pad(x, pad_size):
    x_pad = np.pad((0, 0), (pad_size, pad_size), (pad_size, pad_size), (0, 0), constant_values=0)

    return x_pad


def single_convolution_forward(input_slice, single_filter_w, single_b):
    z_element = input_slice * single_filter_w
    z_element = np.sum(z_element)
    z_element += single_b
    z_element = float(z_element)

    return z_element


def convolution_forward(a, w, b, hparameters):
    (m, input_height, input_width, input_channel) = a.shape
    (filter_size, filter_size, input_channel, output_channel) = w.shape

    pad_size = hparameters['pad_size']
    a = zero_pad(a, pad_size)

    filter_stride = hparameters['filter_stride']
    output_height = int((input_height + 2 * pad_size - filter_size) / filter_stride) + 1
    output_width = int((input_width + 2 * pad_size - filter_size) / filter_stride) + 1

    z = np.zeros((m, output_height, output_width, output_channel))
    for i in range(m):
        for h in range(output_height):
            for w in range(output_width):
                for c in range(output_channel):
                    height_start = h * filter_stride
                    width_start = w * filter_stride
                    height_end = height_start + filter_size
                    width_end = width_start + filter_size

                    a_slice = a[:, height_start:height_end, width_start:width_end, :]

                    z[i, h, w, c] = single_convolution_forward(a_slice, w[:, :, :, c], b[:, :, :, c])

    convolution_cache = (w, b, a)

    return z, convolution_cache


def leaky_relu(z):
    a = np.maximum(0.01 * z, z)
    active_cache = z

    return a, active_cache


def pool_forward(a, hparameters, mode='max'):
    (m, input_height, input_width, input_channel) = a.shape

    pool_size = hparameters['pool_size']
    pool_stride = hparameters['pool_stride']

    output_height = int((input_height - pool_size) / pool_stride + 1)
    output_width = int((input_width - pool_size) / pool_stride + 1)

    pool_a = np.zeros((m, output_height, output_width, input_channel))
    for i in range(m):
        for h in range(output_height):
            for w in range(output_width):
                for c in range(input_channel):
                    height_start = h * pool_stride
                    width_start = w * pool_stride
                    height_end = height_start + pool_size
                    width_end = width_start + pool_size

                    a_slice = a[:, height_start:height_end, width_start:width_end, :]

                    pool_a[i, h, w, c] = np.max(a_slice)

    pool_cache = a
    return pool_a, pool_cache


def single_pool_backward(da, pool_cache):
    max = np.max(pool_cache)
    mask = (max == pool_cache)
    da = da * mask

    return da


def pool_backward(d_pool_a, pool_cache, pool_size, pool_stride):
    (m, pool_a_height, pool_a_width, pool_a_channel) = d_pool_a.shape
    (m, a_height, a_width, pool_a_channel) = pool_cache.shape

    da = np.zeros((m, a_height, a_width, pool_a_channel))

    for i in range(m):
        for h in range(a_height):
            for w in range(a_width):
                for c in range(pool_a_channel):
                    height_start = h * pool_stride
                    width_start = w * pool_stride
                    height_end = height_start + pool_size
                    width_end = width_start + pool_size

                    pool_a_slice = pool_cache[:, height_start:height_end, width_start:width_end, :]

                    da[i, height_start:height_end, width_start:width_end, c] = single_pool_backward(
                        d_pool_a[i, h, w, c], pool_a_slice)

    return da


def leaky_relu_backward(da, active_cache):
    dz = np.ones(da.shape)
    dz[active_cache < 0] = 0.01
    dz = dz * da

    return dz


def convolution_backward(dz, convolution_cache, filter_size, filter_stride, pad_size):
    (m, dz_height, dz_width, dz_channel) = dz.shape
    w, b, a = convolution_cache

    a = zero_pad(a, pad_size)

    (m, a_height, a_width, a_channel) = a.shape
    (filter_size, filter_size, _, _) = w.shape

    no_pad_da = np.zeros((m, a_height - 2 * pad_size, a_width - 2 * pad_size, a_channel))
    da = np.zeros((m, a_height, a_width, a_channel))
    dw = np.zeros((filter_size, filter_size, a_channel, dz_channel))
    db = np.zeros((1, 1, 1, dz_channel))

    for i in range(m):
        for h in range(dz_height):
            for w in range(dz_width):
                for c in range(dz_channel):
                    height_start = h * filter_stride
                    width_start = w * filter_stride
                    height_end = height_start + filter_size
                    width_end = width_start + filter_size

                    a_slice = a[:, height_start:height_end, width_start:width_end, :]

                    dw[:, :, :, c] = dz[i, h, w, c] * a_slice
                    db[:, :, :, c] = dz[i, h, w, c]
                    da[i, height_start:height_end, width_start:width_end, :] = dz[i, h, w, c] * w[:, :, :, c]

        no_pad_da[i, :, :, :] = da[i, pad_size:-pad_size, pad_size:-pad_size, :]

    return dw, db, no_pad_da


def compute_cost():
    return


def optimize(params, hparmas, x, y, num_of_layser, num_of_iteration, learning_rate, lam, batch_size, cost_function,
             pool_size, pool_stride, filter_size, filter_stride, pad_size):
    mini_batches = data_utill.generate_random_mini_batches(x, y, batch_size)
    a = x

    for mini_batch in mini_batches:
        mini_batch_x, mini_batch_y = mini_batch
        for i in range(1, num_of_iteration):
            w = params['w' + str(i)]
            b = params['b' + str(i)]
            z, convolution_cache = convolution_forward(a, w, b, hparmas)
            a, active_cache = leaky_relu(z)
            pool_a, pool_cache = pool_forward(a, hparmas)
            cost = compute_cost()

            da = pool_backward(pool_a, pool_cache, pool_size, pool_stride)
            dz = leaky_relu_backward(da, active_cache)
            dw, db, no_pad_da = convolution_backward(dz, convolution_cache, filter_size, filter_stride, pad_size)

    params, costs = dnn_utill.optimize(params, pool_a, y, num_of_layser, num_of_iteration, learning_rate, lam,
                                       batch_size, cost_function)

    return params, costs
