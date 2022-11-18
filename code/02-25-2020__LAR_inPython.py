'''
File name      : 03-25-2020__LAR_inPython.py
Author         : Jared Hansen
Date created   : 03/25/2020
Python version : 3.7.3

DESCRIPTION:
Coding up LAR-selected LASSO from scratch
Using someone else's LAR code found here
https://github.com/hughperkins/selfstudy-LARS/blob/master/test_lars.ipynb
when I get stuck.
             
'''



## import matplotlib.pyplot as plt
import numpy as np


# Lets imagine a dataset with 2 features, and 2 data points

# Not normalizing mean to 0 for now, otherwise we just can have two different covariant vectors, ie
# (sqrt(2), -sqrt(2)), and (-sqrt(2), sqrt(2))

class ToyData(object):
    def __init__(self):
        pass

toy = ToyData()
toy.m = 2
toy.n = 2
toy.rand = np.random.mtrand.RandomState(132)
toy.X = toy.rand.randn(toy.n, toy.m)
  
print('toy.X', toy.X)
toy.rand = np.random.mtrand.RandomState(122)
toy.y = toy.rand.randint(0, 2, size=(toy.n,))
print('y', toy.y)
print('variate lengths', np.sqrt(np.sum(toy.X * toy.X, 0)))
toy.X /= np.sqrt(np.sum(toy.X * toy.X, 0))
print('variate lengths', np.sqrt(np.sum(toy.X * toy.X, 0)))
print('toy.X', toy.X)
#     toy.X -= np.average(toy.X, 0)
print('toy.X', toy.X)
print('variate lengths', np.sqrt(np.sum(toy.X * toy.X, 0)))
toy.X /= np.sqrt(np.sum(toy.X * toy.X, 0))
print('variate lengths', np.sqrt(np.sum(toy.X * toy.X, 0)))
print('toy.X', toy.X)

toy.residual = toy.y
toy.cos_theta = toy.X.T.dot(toy.residual)
toy.angle = np.arccos(toy.cos_theta) * 180 / 3.1416
print('cos_theta', toy.cos_theta)
print('angle', toy.angle)

plt.ylim(-1.1, 1.1)
plt.xlim(-1.1, 1.1)
plt.xlabel('example 0')
plt.ylabel('example 1')
plt.scatter(toy.X[0], toy.X[1])
plt.scatter(toy.y[0], toy.y[1], color='red')
plt.show()

# toy_stepone()








plt.ylim(-1.1, 1.1)
plt.xlim(-1.1, 1.1)
plt.xlabel('example 0')
plt.ylabel('example 1')
plt.scatter(toy.X[0], toy.X[1], color='blue')
plt.scatter(-toy.X[0], -toy.X[1], color='green')
plt.scatter(toy.y[0], toy.y[1], color='red')
plt.show()

print('angle', toy.angle)
print('angle', 180 - toy.angle)







toy.walk0 = toy.X[:, 1]
toy.walk1 = toy.X[:, 1] - toy.X[:, 0]
# normalize length to 1:
toy.walk1 /= np.sqrt(np.sum(toy.walk1 * toy.walk1))

plt.ylim(-1.1, 1.1)
plt.xlim(-1.1, 1.1)
plt.xlabel('example 0')
plt.ylabel('example 1')
plt.plot((0, toy.walk0[0]), (0, toy.walk0[1]), color='blue')
plt.plot((0, - toy.X[0, 0]), (0, -toy.X[1, 0]), color='green')
plt.plot((0, toy.walk1[0]), (0, toy.walk1[1]), color='purple')
# plt.scatter(-toy.X[0], -toy.X[1], color='green')
plt.scatter(toy.y[0], toy.y[1], color='red')
plt.show()










import numpy as np

toy.sa = np.zeros((2, 2), dtype=np.float32)
toy.sa[:, 0] = toy.walk0
toy.sa[:, 1] = toy.walk1
toy.sb = toy.y
toy.sx = np.linalg.solve(toy.sa, toy.sb)

print(toy.sx)


print(toy.sx[0] * toy.walk0 + toy.sx[1] * toy.walk1)





plt.ylim(-1.1, 1.1)
plt.xlim(-1.1, 1.1)
plt.xlabel('example 0')
plt.ylabel('example 1')

toy.p0 = np.zeros((2,))
toy.p1 = toy.p0 + toy.sx[0] * toy.walk0
toy.p2 = toy.p1 + toy.sx[1] * toy.walk1

plt.plot((toy.p0[0], toy.p1[0]), (toy.p0[1], toy.p1[1]), color='blue')
plt.plot((toy.p1[0], toy.p2[0]), (toy.p1[1], toy.p2[1]), color='purple')
plt.scatter(toy.y[0], toy.y[1], color='red')
plt.show()













import numpy as np
import sklearn.datasets
import matplotlib.pyplot as plt


diabetes_train_splitsize = 1.0

beta_path = None
sum_abs_coeff = None


cement = cement

def fetch_diabetes(subset='train'):
    diabetes = sklearn.datasets.load_diabetes()
    X_all = np.array(cement[['aluminate','ferrite','dicalcium','ferrite']])
    #X_all = diabetes.data
    y_all = np.array(cement['heat'])
    #y_all = diabetes.target

    total_N = len(y_all)
    train_N = int(total_N * diabetes_train_splitsize)
    test_N = total_N - train_N
    rand = np.random.mtrand.RandomState(seed=123)
    train_idx = set(rand.choice(total_N, size=(train_N,), replace=False))

    train_X = X_all[list(train_idx)]
    train_y = y_all[list(train_idx)]

    test_idx = np.zeros((test_N,), dtype=np.int32)
    test_n = 0
    for n in range(total_N):
        if n not in train_idx:
            test_idx[test_n] = n
            test_n += 1
    test_X = X_all[test_idx]
    test_y = y_all[test_idx]

    def get_add_mul(X):
        add = - np.average(X, 0)
        X1 = X + add
        mul = 1 / np.sqrt((X1 * X1).sum(0))
        return add, mul

    X_add, X_mul = get_add_mul(train_X)
    y_add = - np.average(train_y)

    train_X = (train_X + X_add) * X_mul
    train_y = train_y + y_add
    if len(test_X) > 0:
        test_X = (test_X + X_add) * X_mul
        test_y = test_y + y_add

    if subset == 'train':
        return sklearn.datasets.base.Bunch(data=train_X, target=train_y)
    elif subset == 'test':
        return sklearn.datasets.base.Bunch(data=test_X, target=test_y)
    else:
        raise Exception('unknown subset %s' % subset)


def vector_len(vector):
    return np.sqrt(np.sum(vector * vector))


def run_lars(train):
    global beta_path, sum_abs_coeff

    X = train.data
    y = train.target
    m = len(X[0])
    n = len(X)

    active_set = set()
    cur_pred = np.zeros((n,), dtype=np.float32)
    residual = y - cur_pred
    cur_corr = X.transpose().dot(residual)
    j = np.argmax(np.abs(cur_corr), 0)
    print('j', j)
    active_set.add(j)
    beta = np.zeros((m,), dtype=np.float32)
    sign = np.zeros((m,), dtype=np.int32)
    sign[j] = 1

    beta_path = np.zeros((m, m), dtype=np.float32)
    for it in range(m):
        residual = y - cur_pred
        mse = np.sqrt(np.sum(residual * residual))
        print('mse', mse)

        pred_from_beta = X.dot(beta)

        cur_corr = X.transpose().dot(residual)

        X_a = X[:, list(active_set)]
        X_a *= sign[list(active_set)]
        G_a = X_a.transpose().dot(X_a)
        G_a_inv = np.linalg.pinv(G_a)
        G_a_inv_red_cols = np.sum(G_a_inv, 1)
        A_a = 1 / np.sqrt(np.sum(G_a_inv_red_cols))
        omega = A_a * G_a_inv_red_cols
        equiangular = X_a.dot(omega)  # .reshape(n)

        cos_angle = X.transpose().dot(equiangular)
        gamma = None
        largest_abs_correlation = np.abs(cur_corr).max()
        print('largest_abs_correlation', largest_abs_correlation)
        if it < m - 1:
            next_j = None
            next_sign = 0
            for j in range(m):
                if j in active_set:
                    continue
                v0 = (largest_abs_correlation - cur_corr[j]) / (A_a - cos_angle[j]).item()
                v1 = (largest_abs_correlation + cur_corr[j]) / (A_a + cos_angle[j]).item()
                if v0 > 0 and (gamma is None or v0 < gamma):
                    next_j = j
                    gamma = v0
                    next_sign = 1
                if v1 > 0 and (gamma is None or v1 < gamma):
                    gamma = v1
                    next_j = j
                    next_sign = -1
        else:
            gamma = largest_abs_correlation / A_a

        # coeffs, eg for 3 vectors
        #
        # I reckon there should be a faster (ie closed-form) way of doing this, but this way works for now:
        #
        # c_0 * x_0 + c_1 * x_1 + c_2 * x_2 = equiangular
        # we know: x_0, x_1, x_2, equiangular
        # they're all vectors, so equation must hold for each dimension
        # ie, we have equations:
        # c_0 * x_0_0 + c_1 * x_1_0 + c_2 * x_2_0 = equi_0
        # c_0 * x_0_1 + c_1 * x_1_1 + c_2 * x_2_1 = equi_1
        # c_0 * x_0_2 + c_1 * x_1_2 + c_2 * x_2_2 = equi_2
        # ...
        # comparing with doc at
        # https://docs.scipy.org/doc/numpy/reference/generated/numpy.linalg.lstsq.html#numpy.linalg.lstsq
        # their x_0, x_1, x_2 is our c_0, c_1, c_2
        # their constants are our x_0_0, ...
        sa = X_a
        sb = equiangular * gamma
        sx = np.linalg.lstsq(sa, sb)
        for i, j in enumerate(active_set):
            beta[j] += sx[0][i] * sign[j]

        print('next j', next_j, 'next sign', next_sign, 'gamma', gamma, 'new max correlation: %s' % (
            largest_abs_correlation - gamma * A_a))

        cur_pred = X.dot(beta)
        active_set.add(next_j)
        sign[next_j] = next_sign

        beta_path[it, :] = beta
        print('beta', beta)

    print('beta_path', beta_path)
    sum_abs_coeff = np.sum(np.abs(beta_path), 1)
    print('sum_abs_coeff', sum_abs_coeff)
    

def run():
    train = fetch_diabetes(subset='train')
    test = fetch_diabetes(subset='test')
    print((train.data * train.data).sum(0))
    print((test.data * test.data).sum(0))
    print(np.average(train.target, 0))
    print(np.average(test.target, 0))
    run_lars(train)


run()




# plotting code is based on:
# http://scikit-learn.org/stable/auto_examples/linear_model/plot_lasso_lars.html
plt.plot(sum_abs_coeff, beta_path)
plt.title('LARS Path')
plt.ylabel('beta_j')
plt.xlabel('sum_j(|coeff_j|)')
plt.show()




























































