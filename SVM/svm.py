from numpy import *


def load_data_set(filename):
    data_mat = []
    label_mat = []
    with open(filename) as fr:
        for line in fr.readlines():
            line_arr = line.strip().split('\t')
            data_mat.append([float(line_arr[0]), float(line_arr[1])])
            label_mat.append(float(line_arr[2]))
    return data_mat, label_mat


def select_j_rand(i, m):
    j = i
    while (j == i):
        j = int(random.uniform(0, m))
    return j


def clip_alpha(aj, H, L):
    if aj > H:
        aj = H
    if L > aj:
        aj = L
    return aj


def smo_simple(data_mat_in, label_mat_in, c, toler, max_iter):
    data_mat = mat(data_mat_in)
    label_mat = mat(label_mat_in).transpose()
    b = 0.0
    m, n = shape(data_mat)
    iter = 0
    alpha = mat(zeros((m, 1)))
    while(iter < max_iter):
        alpha_pairs_changed = 0
        for i in range(m):
            yi = float(multiply(alpha, label_mat).T *
                       (data_mat * data_mat[i, :].T) + b)
            Ei = yi - float(label_mat[i])
            if (((label_mat[i] * Ei < -toler) and (alpha[i] < c)) or
                    ((label_mat[i] * Ei > toler) and (alpha[i] > 0))):
                j = select_j_rand(i, m)
                yj = float(multiply(alpha, label_mat).T *
                           (data_mat * data_mat[j, :].T) + b)
                Ej = yj - float(label_mat[j])
                alphai_old = alpha[i].copy()
                alphaj_old = alpha[j].copy()
                if label_mat[i] != label_mat[j]:
                    L = max(0, alphaj_old - alphai_old)
                    H = min(c, c + alphaj_old - alphai_old)
                else:
                    L = max(0, alphaj_old + alphai_old - c)
                    H = min(c, alphaj_old + alphai_old)
                if L == H:
                    print('L==H')
                    continue
                eta = (-data_mat[i, :] * data_mat[i, :].T - data_mat[j, :] *
                       data_mat[j, :].T + 2.0 * data_mat[i, :] *
                       data_mat[j, :].T)
                if eta >= 0:
                    print('eta>=0')
                    continue
                alpha[j] -= label_mat[j] * (Ei - Ej) / eta
                alpha[j] = clip_alpha(alpha[j], H, L)
                if abs(alpha[j] - alphaj_old) < 0.00001:
                    print('j not moving enough')
                    continue
                alpha[i] += (label_mat[i] * label_mat[j] *
                             (alphaj_old - alpha[j]))
                b1 = (b - Ei - label_mat[i] * (alpha[i] - alphai_old) *
                      (data_mat[i, :] * data_mat[i, :].T) -
                      label_mat[j] * (alpha[j] - alphaj_old) *
                      (data_mat[i, :] * data_mat[j, :].T))
                b2 = (b - Ej - label_mat[i] * (alpha[i] - alphai_old) *
                      (data_mat[i, :] * data_mat[j, :].T) -
                      label_mat[j] * (alpha[j] - alphaj_old) *
                      (data_mat[j, :] * data_mat[j, :].T))
                if (alpha[i] < c) and (alpha[i] > 0):
                    b = b1
                elif (alpha[j] < c) and (alpha[j] > 0):
                    b = b2
                else:
                    b = (b1 + b2) / 2.0
                alpha_pairs_changed += 1
                print('iter: %d i: %d, pairs changed %d' %
                      (iter, i, alpha_pairs_changed))
        if (alpha_pairs_changed == 0):
            iter += 1
        else:
            iter = 0
        print('iter number: %d' % iter)
    return b, alpha


def plot_best_fit(filename, b, alpha):
    import matplotlib.pyplot as plt
    data_mat_in, label_mat_in = load_data_set(filename)
    data_mat = mat(data_mat_in)
    label_mat = mat(label_mat_in).transpose()
    n = shape(data_mat)[0]
    xcord1 = []
    ycord1 = []
    xcord2 = []
    ycord2 = []
    for i in range(n):
        if label_mat[i] == 1:
            xcord1.append(data_mat[i, 0])
            ycord1.append(data_mat[i, 1])
        else:
            xcord2.append(data_mat[i, 0])
            ycord2.append(data_mat[i, 1])
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.scatter(xcord1, ycord1, s=30, c='red', marker='s')
    ax.scatter(xcord2, ycord2, s=30, c='green')
    weights = (multiply(alpha, label_mat).T * data_mat).T
    weights = weights.getA()
    y = arange(-6.0, 4.0, 0.1)
    x = -(weights[1] * y + b[0]) / weights[0]
    ax.plot(x, y)
    plt.show()

data_arr, label_arr = load_data_set('testSet.txt')
b, alpha = smo_simple(data_arr, label_arr, 0.6, 0.001, 40)
print(b)
plot_best_fit('testSet.txt', b.getA(), alpha)
