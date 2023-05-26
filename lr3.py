from scipy.optimize import linprog


# способ №1
obj = [-130.5, -20, -56, -87.8]

lhs_ineq = [[6, -2, -4, 1],
            [4, -1.5, 10.4, 13]]

rhs_ineq = [-450, 89]

lhs_eq = [[-1.8, 2, 1, -4]]
rhs_eq = [756]

bnd = [(0, float('inf')),
       (0, float('inf')),
       (0, float('inf')),
       (0, float('inf'))]

opt = linprog(c=obj, A_ub=lhs_ineq, b_ub=rhs_ineq,
              A_eq=lhs_eq, b_eq=rhs_eq, bounds=bnd, method='highs')

print(opt)


# способ №2
import numpy as np
import warnings
warnings.filterwarnings('ignore')

a = np.array([80, 60, 100])
b = np.array([40, 60, 40, 50, 50])

D = np.array([[6, 4, 3, 4, 2],
             [3, 6, 4, 9, 2],
             [3, 1, 2, 2, 6]])


def delta(a, b, c, x):
    if a.sum() > b.sum:
        b = np.hstack((b, [a.sum() - b.sum()]))
        c = np.hstack((c, np.zeros(len(a)).reshape(-1, 1)))
    elif a.sum() < b.sum():
        a = np.hstack((a, [b.sum() - a.sum()]))
        c = np.hstack((c, np.zeros(len(b))))

    m = len(a)
    n = len(b)

    u = np.zeros(m)
    v = np.zeros(n)

    for i in range(m):
        for j in range(n):
            if x[i, j] != 0:
                if v[j] != 0:
                    u[i] = c[i, j] - v[j]
                else:
                    v[j] = c[i, j] - u[i]

    delta = np.zeros((m, n))
    for i in range(m):
        for j in range(n):
            delta[i, j] = u[i] + v[j] - c[i, j]

    return delta


def prepare(a, b):
    m = len(a)
    n = len(b)
    h = np.diag(np.ones(n))
    v = np.zeros((m, n))
    v[0] = 1
    for i in range(1, m):
        h = np.hstack((h, np.diag(np.ones(n))))
        k = np.zeros((m, n))
        k[i] = 1
        v = np.hstack((v, k))
    return np.vstack((h, v)).astype(int), np.hstack((b, a))


def potenz(a_, b_, c_):
    a = np.copy(a_)
    b = np.copy(b_)
    c = np.copy(c_)

    if a.sum() > b.sum:
        b = np.hstack((b, [a.sum() - b.sum()]))
        c = np.hstack((c, np.zeros(len(a)).reshape(-1, 1)))
    elif a.sum() < b.sum():
        a = np.hstack((a, [b.sum() - a.sum()]))
        c = np.hstack((c, np.zeros(len(b))))

    m = len(a)
    n = len(b)

    A_eq, b_eq = prepare(a, b)
    res = linprog(c.reshape(1, -1), A_eq=A_eq, b_eq=b_eq, bounds=(0, None), method='simplex')
