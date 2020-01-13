import numpy as np
import matplotlib.pyplot as plt
import supervised_learning_ml_training_server.utils.logistic_regression_util as logistic_regression_util
import data.data_utils as data_utils


def build_logistic_regression_model(x_train, y_train, x_test, y_test, learning_rate, num_of_iterations):
    dim = x_train.shape[0]
    params = logistic_regression_util.initialize_parameters(dim)
    w = params["w"]
    b = params["b"]

    params, costs = logistic_regression_util.optimize(w, b, x_train, y_train, learning_rate, num_of_iterations)

    train_prediction = logistic_regression_util.predict(params, x_train)
    test_prediction = logistic_regression_util.predict(params, x_test)
    print("train accuracy: {}%".format(100 - np.mean(np.abs(train_prediction - y_train)) * 100))
    print("test accuracy: {}%".format(100 - np.mean(np.abs(test_prediction - y_test)) * 100))

    plt.figure()
    plt.plot(costs)
    plt.xlabel("logistic regression")
    plt.ylabel("cost")
    plt.show()

    return params


# logistic regression 모델은 개와 고양이를 구분하는 모델입니다
# 아래에서 사용한 데이터는 손가락 숫자 0, 1, 2, 3, 4, 5 rgb 이미지들입니다
# 예측한 결과 벡터 y'의 모양을 잘 생각해보고, logistic regression 모델로 아래의 데이터를 분석할 수 없는 이유를 설명하세요
# 2진 분류 데이터를 구글에서 구한 후, data_utils 모듈에 적당한 함수를 구현하고 모델을 올바르게 사용해보세요
# 데이터는 cat vs dog를 추천합니다
# 트레이닝이 끝난 후, 자신만의 개와 고양이 사진을 구글링하고, 실제로 잘 분류되는지 검증해보세요
x_train, y_train, x_test, y_test, _ = data_utils.load_sign_dataset()

x_train, y_train = data_utils.flatten_and_reshape(x_train, y_train)
x_test, y_test = data_utils.flatten_and_reshape(x_test, y_test)

x_train = data_utils.centralize(x_train)
x_test = data_utils.centralize(x_test)

params = build_logistic_regression_model(x_train, y_train, x_test, y_test, 0.005, 5000)
