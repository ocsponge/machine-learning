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


def stand_regres(xarr, yarr):
    xmat = mat(xarr)
    ymat = mat(yarr).T
    xTx = xmat.T * xmat
    if linalg.det(xTx) == 0:
        print('this matrix is singular, cannot do inverse')
        return
    ws = xTx.I * xmat.T * ymat
    return ws


def lwlr(test_point, xarr, yarr, k=1.0):
    xmat = mat(xarr)
    ymat = mat(yarr).T
    m = shape(xmat)[0]
    w = mat(eye((m)))
    for i in range(m):
        diff = test_point - xmat[i, :]
        w[i, i] = exp(diff * diff.T / (-2.0 * k**2))
    xTx = xmat.T * w * xmat
    if linalg.det(xTx) == 0:
        print('this matrix is singular, cannot do inverse')
        return
    ws = xTx.I * xmat.T * w * ymat
    return test_point * ws


def lwlr_test(test_arr, xarr, yarr, k=1.0):
    m = shape(test_arr)[0]
    yhat = zeros(m)
    for i in range(m):
        yhat[i] = lwlr(test_arr[i], xarr, yarr, k)
    return yhat


def error(yarr, yhat_arr):
    return ((yarr - yhat_arr)**2).sum()


'''xarr, yarr = load_data_set('ex0.txt')
yhat = lwlr_test(xarr, xarr, yarr, 0.003)
xmat = mat(xarr)
ymat = mat(yarr)
fig = plt.figure()
ax = fig.add_subplot(111)
ax.scatter(xmat[:, 1], ymat.T, s=2, c='red')
xcopy = xmat.copy()
xcopy.sort(0)
id = xmat[:, 1].argsort(0)
ax.plot(xcopy[:, 1], yhat[id])
plt.show()'''

abx, aby = load_data_set('abalone.txt')
yhat01 = lwlr_test(abx[100:199], abx[0:99], aby[0:99], 0.1)
yhat1 = lwlr_test(abx[100:199], abx[0:99], aby[0:99], 1)
yhat10 = lwlr_test(abx[100:199], abx[0:99], aby[0:99], 10)
e01 = error(aby[100:199], yhat01.T)
e1 = error(aby[100:199], yhat1.T)
e10 = error(aby[100:199], yhat10.T)
print(e01)
print(e1)
print(e10)
ws = stand_regres(abx[0:99], aby[0:99])
yhat = mat(abx[100:199]) * ws
er = error(aby[100:199], yhat.T.A)
print(er)
