import matplotlib.pyplot as plt
import numpy as np
import supercised_learning.utills.dnn_utill as dnn_utill
import data.data_utill as data_utill


def build_dnn_model(x_train, y_train, x_test, y_test, layer_dim, learning_rate, num_of_iterations, lam, batch_size,
                    cost_function):
    num_of_layers = len(layer_dim)

    params = dnn_utill.initialize_parameter(layer_dim)

    params, costs = dnn_utill.optimize(params, x_train, y_train, num_of_layers, num_of_iterations, learning_rate, lam,
                                       batch_size, cost_function)

    prediction_train = dnn_utill.predict(params, x_train, num_of_layers)
    prediction_test = dnn_utill.predict(params, x_test, num_of_layers)

    prediction_train_accuracy = 100 - np.mean(np.abs(y_train - prediction_train)) * 100
    prediction_test_accuracy = 100 - np.mean(np.abs(y_test - prediction_test)) * 100

    print('prediction_train_accuracy : ', prediction_train_accuracy)
    print('prediction_test_accuracy : ', prediction_test_accuracy)

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

layer_dim = [dim, 64, 64, output_dim]

build_dnn_model(x_train, y_train, x_test, y_test, layer_dim, learning_rate=0.1, num_of_iterations=2000, lam=0,
                batch_size=108, cost_function='cross_entropy')
