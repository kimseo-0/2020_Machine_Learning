import numpy as np


def initialize_parameters(dim):
    params = {}

    params["w"] = np.random.randn(dim, 1) * 0.01
    params["b"] = 0

    return params


def sigmoid(z):
    a = 1 / (1 + np.exp(-z))

    return a


def forward_and_backward(w, b, x, y, lam):
    m = x.shape[1]
    #
    a = sigmoid(np.matmul(w.T, x) + b)

    # 머신러닝에서 많이 쓰이는 cost 함수는 mean square error, cross entropy 2가지가 있다
    # 2가지의 cost 함수에 각각에 대한 dw, db를 직접 계산하고 구현한다
    # 2가지의 cost 함수를 각각 사용해본다

    # mean square error를 사용하는 경우의 cost
    # 서영찡의 코드
    cost_square = (np.sum((a - y) * (a - y)) + lam * np.sum(w * w)) / (2 * m)
    cost_square = float(cost_square)

    # mean square error를 사용하는 경우의 dw, db
    # 서영찡의 코드
    dw_squre = ((np.matmul(np.matmul(a, (a - y).T)), (1 - a).T) + lam * w) / m
    db_squre = (np.sum(a - y) + lam * w) / m
    db_squre = float(db_squre)

    # cross entropy를 사용하는 경우의 cost
    cost = np.sum(-(y * np.log(a) + (1 - y) * np.log(1 - a))) / m + np.sum(w * w) / (2 * m)
    cost = float(cost)

    # cross entropy를 사용하는 경우의 dw, db
    dw = (np.matmul(x, (a - y).T) + w) / m
    db = np.sum((a - y)) / m
    db = float(db)

    grads = {"dw": dw,
             "db": db}

    return grads, cost


def optimize(w, b, x, y, learning_rate, num_of_iterations, lam):
    costs = []

    for i in range(num_of_iterations):
        grads, cost = forward_and_backward(w, b, x, y, lam)

        dw = grads["dw"]
        db = grads["db"]

        w = w - learning_rate * dw
        b = b - learning_rate * db

        if i % 100 == 0:
            costs.append(cost)
            print("cost after iteration %d: %f" % (i, cost))

    params = {"w": w,
              "b": b}

    return params, costs


def predict(params, test_input):
    m = test_input.shape[1]

    w = params["w"]
    b = params["b"]
    a = sigmoid(np.matmul(w.T, test_input) + b)

    prediction = np.zeros((1, m))
    for i in range(m):
        if a[0, i] > 0.75:
            prediction[0, i] = 1

    return prediction
