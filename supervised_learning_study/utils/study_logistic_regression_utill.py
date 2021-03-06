import numpy as np


def initaialize_parameters(dim):
    '''
    초기 w, b값 설정
    :param dim: data의 차원
    :return: 초기화된 w,b값이 저장되어 있는 dict = {'w' : w, 'b' : b}
    '''
    params = {}
    w = np.random.randn(dim, 1) * 0.01  # (dim, 1)의 matrix, value = 가우시안
    b = 0

    params['w'] = w
    params['b'] = b

    return params


def linear_forward(w, b, a):
    '''

    :param w: [[]] (data dim, 1) -> w.T (1, data dim)
    :param b: 상수
    :param a: 데이터 [[]] (data dim, m)
    :return: 선형 변화 결과 z, linear_cache = w, b, a
    '''
    linear_cache = w, b, a
    z = np.matmul(w.T, a) + b  # w는 열벡터이기 때문에 행벡터로 변형

    return z, linear_cache


def sigmoid(z):
    '''
    비선형 변환 - sigmoid
    :param z: 선형 변환 결과 z
    :return: 비선형 변환 결과 a, active_cache = z
    '''
    active_cache = z
    a = 1 / (1 + np.exp(-z))

    return a, active_cache


def linear_activation_forward(w, b, a, activation):
    '''
    선형 변환 & 비선형 변환
    :param w: [[]]
    :param b: 상수
    :param a: [[]]
    :param activation: 비선형 변환 할 함수 이름 (sigmoid, relu, reaky_rlelu, softmax)
    :return: a, linear_activation_cache
    '''
    z, linear_cache = linear_forward(w, b, a)

    if activation == 'sigmoid':
        a, active_cache = sigmoid(z)

    linear_activation_cache = (linear_cache, active_cache)

    return a, linear_activation_cache


def forward_and_backward(w, b, x, y, lam):
    '''
    선형, 비선형 변환
    calculate_cost
    backward_propagation
    :param w: [[]]
    :param b: 상수
    :param x: 데이터 [[]...]
    :param y: 정답 데이터
    :param lam:
    :return: grad = {'dw' : dw, 'db' : db}, costs = [cost]
    '''
    m = x.shape[1]  # num of data sets

    # 선형, 비선형 변환
    a, linear_activation_cache = linear_activation_forward(w, b, x, 'sigmoid')

    # cost 구하기
    cost = np.sum(- y * np.log(a) + (1 - y) * np.log(1 - a), axis=1) / m + lam * np.sum(w * w) / (2 * m)
    cost = float(cost)

    # dw, db 구하기
    grads = {}
    dw = (np.matmul(x, (a - y).T) + lam * w) / m
    db = np.sum(a - y) / m
    db = float(db)

    grads['dw'] = dw
    grads['db'] = db

    return grads, cost


def optimize(w, b, x, y, num_of_iteration, learning_rate, lam):
    '''
    gradient decent 과정
    optimize된 w, b값
    :param w: [[]]
    :param b: 상수
    :param x: 데이터 [[]...data dim]
    :param y: 정답 데이터 [[]]
    :param num_of_iteration: 반복 횟수
    :param learning_rate: 한번 반복할 때 속도 조절(보폭의 크기 조절)
    :param lam:
    :return: w, b
    '''
    params = {}
    costs = []

    for i in range(num_of_iteration):
        grads, cost = forward_and_backward(w, b, x, y, lam)

        dw = grads['dw']
        db = grads['db']

        w = w - learning_rate * dw
        b = b - learning_rate * db

        costs.append(cost)

    params['w'] = w
    params['b'] = b

    return params, costs


def predict(params, test_input):
    m = test_input.shape[1]

    w = params['w']
    b = params['b']

    a = sigmoid(np.matmul(w.T, test_input) + b)

    prediction = np.zeros(1, m)
    for i in range(m):
        if a[0, i] > 0.75:
            prediction[0, i] = 1
        else:
            prediction[0, i] = 0

    return prediction
