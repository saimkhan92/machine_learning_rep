import numpy as np

kINSP = np.array([(1, 8, +1),
               (7, 2, -1),
               (6, -1, -1),
               (-5, 0, +1),
               (-5, 1, -1),
               (-5, 2, +1),
               (6, 3, +1),
               (6, 1, -1),
               (5, 2, -1)])

kSEP = np.array([(-2, 2, +1),    # 0 - A
              (0, 4, +1),     # 1 - B
              (2, 1, +1),     # 2 - C
              (-2, -3, -1),   # 3 - D
              (0, -1, -1),    # 4 - E
              (2, -3, -1),    # 5 - F
              ])


def weight_vector(x, y, alpha):

    w = np.zeros(len(x[0]))
    # TODO: IMPLEMENT THIS FUNCTION
    for i in range(len(x)):
        w += x[i] * y[i] * alpha[i]

    return w

def find_support(x, y, w, b, tolerance=0.001):

    support = set()
    # TODO: IMPLEMENT THIS FUNCTION

    for i in range(len(x)):
        measure = (sum(w*x[i]) + b)*y[i]
        if (measure > 1 - tolerance) and (measure < 1 + tolerance):
            support.add(i)
    return support

def find_slack(x, y, w, b):

    slack = set()
    # TODO: IMPLEMENT THIS FUNCTION
    for i in range(len(x)):
        measure = (sum(w*x[i]) + b)*y[i]
        if (measure <1.0):
            slack.add(i)
    return slack
