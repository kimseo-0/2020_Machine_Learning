import matplotlib.pyplot as plt
import numpy as np
import supercised_learning.utills.logistic_regression_utill as logistic_regression_utill
import data.data_utill as data_utill


def build_logistic_regression_model(x_train, y_train, x_test, y_test, learning_rate, num_of_iterations):
    dim = x_train.shape(0)

    params = logistic_regression_utill.initialize_parameters(dim)
    w = params['w']
    b = params['b']

    params, costs = logistic_regression_utill.optimize(w, b, x_train, y_train, num_of_iterations, learning_rate)

    prediction_train = logistic_regression_utill.predict(params, x_train)
    prediction_test = logistic_regression_utill.predict(params, x_test)

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


x_train, y_train, x_test, y_test, dim = data_utill.load_sign_dataset()

x_train, y_train = data_utill.flatten_and_reshape(x_train, y_train)
x_test, y_test = data_utill.flatten_and_reshape(x_test, y_test)

x_train = data_utill.cetralize(x_train)
x_test = data_utill.cetralize(x_test)

build_logistic_regression_model(x_train, y_train, x_test, y_test, 0.05, 5000)
