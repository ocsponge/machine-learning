from numpy import *
import matplotlib.pyplot as plt


def load_data_set(filename):
    with open(filename) as ff:
        num_feature = len(ff.readline().split('\t'))
    data_mat = []
    label_mat = []
    with open(filename) as fr:
        for line in fr.readlines():
            one_line = []
            line_arr = line.strip().split('\t')
            for i in range(num_feature - 1):
                one_line.append(float(line_arr[i]))
            data_mat.append(one_line)
            label_mat.append(float(line_arr[-1]))
    return data_mat, label_mat


def ridge_regres(xmat, ymat, lamda=0.2):
    xTx = xmat.T * xmat
    denom = xTx + eye(shape(xmat)[1]) * lamda
    if linalg.det(denom) == 0:
        print('this matrix is singular, cannot do inverse')
        return
    ws = denom.I * xmat.T * ymat
    return ws


def ridge_test(xarr, yarr):
    xmat = mat(xarr)
    ymat = mat(yarr).T
    ymean = mean(ymat, 0)
    ymat = ymat - ymean
    xmean = mean(xmat, 0)
    xvar = var(xmat, 0)
    xmat = (xmat - xmean) / xvar
    num_test = 30
    wmat = zeros((num_test, shape(xmat)[1]))
    for i in range(num_test):
        ws = ridge_regres(xmat, ymat, exp(i - 10))
        wmat[i, :] = ws.T
    return wmat


def regularize(xmat):
    inxmat = xmat.copy()
    xmean = mean(inxmat, 0)
    xvar = var(inxmat, 0)
    inxmat = (inxmat - xmean) / xvar
    return inxmat


def stage_wise(xarr, yarr, eps=0.01, numit=100):
    xmat = mat(xarr)
    ymat = mat(yarr).T
    ymean = mean(ymat, 0)
    ymat = ymat - ymean
    xmat = regularize(xmat)
    m, n = shape(xmat)
    ret_ws = zeros((numit, n))
    ws = zeros((n, 1))
    ws_max = ws.copy()
    for i in range(numit):
        print(ws.T)
        min_error = inf
        for j in range(n):
            for sign in [-1, 1]:
                ws_test = ws.copy()
                ws_test[j, :] += eps * sign
                yhat = xmat * ws_test
                er = error(ymat.A, yhat.A)
                if er < min_error:
                    min_error = er
                    ws_max = ws_test
        ws = ws_max.copy()
        ret_ws[i, :] = ws.T
    return ret_ws


def error(yarr, yhat_arr):
    return ((yarr - yhat_arr)**2).sum()


xarr, yarr = load_data_set('abalone.txt')
ws = stage_wise(xarr, yarr, 0.005, 1000)
fig = plt.figure()
ax = fig.add_subplot(111)
ax.plot(ws)
plt.show()
