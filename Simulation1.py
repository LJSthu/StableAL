import numpy as np
from numpy.random import seed
seed(1)
import random
random.seed(1)
from SAL import StableAL


def sign(x):
    if x > 0:
        return 1
    if x < 0:
        return -1
    return 0


def data_generation(n1, n2, ps, pvb, pv, r):
    S = np.random.normal(0, 1, [n1, ps])
    V = np.random.normal(0, 1, [n1, pvb + pv])

    Z = np.random.normal(0, 1, [n1, ps + 1])
    for i in range(ps):
        S[:, i:i + 1] = 0.8 * Z[:, i:i + 1] + 0.2 * Z[:, i + 1:i + 2]

    beta = np.zeros((ps, 1))
    for i in range(ps):
        beta[i] = (-1) ** i * (i % 3 + 1) * 1.0 / 3

    noise = np.random.normal(0, 0.3, [n1, 1])

    Y = np.dot(S, beta) + noise + 1 * S[:, 0:1] * S[:, 1:2] * S[:, 2:3]
    Y_compare = np.dot(S, beta) + 1 * S[:, 0:1] * S[:, 1:2] * S[:, 2:3]

    index_pre = np.ones([n1, 1], dtype=bool)
    for i in range(pvb):
        D = np.abs(V[:, pv + i:pv + i + 1] * sign(r) - Y_compare)
        pro = np.power(np.abs(r), -D * 5)
        selection_bias = np.random.random([n1, 1])
        index_pre = index_pre & (
                    selection_bias < pro)
    index = np.where(index_pre == True)

    S_re = S[index[0], :]
    V_re = V[index[0], :]
    Y_re = Y[index[0]]


    n, p = S_re.shape

    index_s = np.random.permutation(n)

    X_re = np.hstack((S_re, V_re))
    beta_X = np.vstack((beta, np.zeros((pv + pvb, 1))))

    return X_re[index_s[0:n2], :], Y_re[index_s[0:n2], :], beta_X

def generate_sim1_training_data():
    n1 = 200000
    p = 10
    ps = int(p * 0.5)
    pvb = int(p * 0.1)
    pv = p - ps - pvb

    r_list = [-1.1]
    environments = []

    r = 2.0
    n2 = 3900
    trainx, trainy, real_beta = data_generation(n1, n2, ps, pvb, pv, r)
    environments.append([trainx, trainy])

    n3 = 100
    for r in r_list:
        x_bias, y_bias, real_beta = data_generation(n1, n3, ps, pvb, pv, r)
        environments.append([x_bias, y_bias])

    return environments, real_beta


def generate_sim1_testing_data():
    n1 = 200000
    n2 = 1000
    p = 10
    ps = int(p * 0.5)
    pvb = int(p * 0.1)
    pv = p - ps - pvb
    r_list = [-3, -2.5, -2, -1.7, -1.5, 1.5, 1.7, 2, 2.5, 3]
    X = []
    y = []
    for r in r_list:
        trainx, trainy, real_beta = data_generation(n1, n2, ps, pvb, pv, r)
        X.append(trainx)
        y.append(trainy)
    return X, y


if __name__ == "__main__":
    training_environments, real_beta = generate_sim1_training_data()
    method = StableAL(training_environments)
    method.trainAll(10,100,15)