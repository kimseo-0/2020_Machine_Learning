import tensorflow as tf


def create_placeholders(input_dim, output_dim):
    # None은 training 갯수
    X = tf.placeholder(tf.float32, [input_dim, None], 'x')
    Y = tf.placeholder(tf.float32, [output_dim, None], 'y')

    return X, Y


def initialize_parameters(dims_of_layers):
    params = {}
    for i in range(1, len(dims_of_layers)):
        params['w' + str(i)] = tf.get_variable('w' + str(i), [dims_of_layers[i], dims_of_layers[i - 1]],
                                               initializer=tf.keras.initializers.he_uniform())
        params['b' + str(i)] = tf.get_variable('b' + str(i), [dims_of_layers[i], 1], initializer=tf.zeros_initializer())

    return params


def linear_forward(w, b, x):
    z = tf.matmul(w, x) + b

    return z


def forward_propagation(params, x, dims_of_layers):
    a = x
    for i in range(1, len(dims_of_layers)):
        w = params['w' + str(i)],
        b = params['b' + str(i)]
        z = linear_forward(w, b, a)
        if i == len(dims_of_layers) - 1:
            a = tf.nn.softmax(z, axis=0)
        else:
            a = tf.nn.relu(z)
    return a


def compute_cost(a, y):
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=a, labels=y))

    return cost


def predict(params, x, y, dims_of_layers):
    a = forward_propagation(params, x, dims_of_layers)

    p = tf.equal(tf.argmax(a), tf.argmax(y))

    return p
