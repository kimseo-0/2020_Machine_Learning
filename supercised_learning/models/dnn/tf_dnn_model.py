import matplotlib.pyplot as plt
import numpy as np
import supercised_learning.utills.tf_dnn_utill as tf_dnn_utill
import data.data_utill as data_utill
import tensorflow as tf



def build_tf_dnn_model(x_train, y_train, x_test, y_test, layer_dim, learning_rate, num_of_iterations, lam, batch_size,
                       cost_function):
    x, y = tf_dnn_utill.create_placeholders(layer_dim[0], layer_dim[-1])

    params = tf_dnn_utill.initialize_parameters(layer_dim)

    a = tf_dnn_utill.forward_propagation(params, x, layer_dim)

    cost = tf_dnn_utill.compute_cost(a, y)

    optimizer = tf.train.AdamOptimizer(learning_rate).minimize(cost)

    # 변수 초기화
    init = tf.global_variables_initializer()
    costs = []

    with tf.Session() as sess:
        sess.run(init)

        mini_batches = data_utill.generate_random_mini_batches(x_train, y_train, batch_size)

        for mini_batch in mini_batches:
            mini_batch_x = mini_batch[0]
            mini_batch_y = mini_batch[1]
            for i in range(num_of_iterations):
                _, mini_batch_cost = sess.run([optimizer, cost], {x: mini_batch_x, y: mini_batch_y})

                if i % 100 == 0:
                    print('cost ', i, ' : ', mini_batch_cost)
                    costs.append(mini_batch_cost)

        prediction_train = tf_dnn_utill.predict(params, x_train, y_train, layer_dim)
        prediction_test = tf_dnn_utill.predict(params, x_test, y_test, layer_dim)

        print('prediction_train_accuracy : ', prediction_train.eval(prediction_train, {x: x_train, y: y_train}))
        print('prediction_test_accuracy : ', prediction_train.eval(prediction_test, {x: x_test, y: y_test}))

        plt.figure()
        plt.plot(costs)
        plt.xlabel('num_of_iterations(per 100)')
        plt.ylabel('cost')
        plt.show()

    return params, costs


x_train, y_train, x_test, y_test, dim, output_dim = data_utill.load_sign_dataset()

x_train, y_train = data_utill.flatten_and_reshape(x_train, y_train)
x_test, y_test = data_utill.flatten_and_reshape(x_test, y_test)

x_train = data_utill.cetralize(x_train)
x_test = data_utill.cetralize(x_test)

layer_dim = [dim, 64, 16, output_dim]

build_tf_dnn_model(x_train, y_train, x_test, y_test, layer_dim, learning_rate=0.001, num_of_iterations=5000, lam=0.01,
                   batch_size=32, cost_function='cross_entropy')
