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


def kernel_trans(X, A, ktype):
    m, n = shape(X)
    k = mat(zeros((m, 1)))
    if ktype[0] == 'lin':
        k = X * A.T
    elif ktype[0] == 'rbf':
        for j in range(m):
            delta = X[j, :] - A
            k[j] = delta * delta.T
        k = exp(k / (-1.0 * ktype[1]**2))
    else:
        raise NameError('that kernel is not recognised')
    return k


class data_struct:

    def __init__(self, data_mat, label_mat, c, toler, ktype):
        self.data = data_mat
        self.label = label_mat
        self.c = c
        self.toler = toler
        self.m = shape(data_mat)[0]
        self.b = 0.0
        self.alpha = mat(zeros((self.m, 1)))
        self.Ecache = mat(zeros((self.m, 2)))
        self.k = mat(zeros((self.m, self.m)))
        for i in range(self.m):
            self.k[:, i] = kernel_trans(data_mat, data_mat[i], ktype)


def calcu_Ek(os, k):
    yk = float(multiply(os.label, os.alpha).T * os.k[:, k]) + os.b
    Ek = yk - float(os.label[k])
    return Ek


def select_j(os, i, Ei):
    j = -1
    Ej = 0.0
    max_delta_E = 0.0
    os.Ecache[i] = [1, Ei]
    valid_j = nonzero(os.Ecache[:, 0].A)[0]
    if len(valid_j) > 1:
        for k in valid_j:
            if k == i:
                continue
            Ek = calcu_Ek(os, k)
            delta_E = abs(Ei - Ek)
            if delta_E > max_delta_E:
                j = k
                Ej = Ek
                max_delta_E = delta_E
    else:
        j = select_j_rand(i, os.m)
        Ej = calcu_Ek(os, j)
    return j, Ej


def update_E(os, k):
    Ek = calcu_Ek(os, k)
    os.Ecache[k] = [1, Ek]


def innerL(os, i):
    Ei = calcu_Ek(os, i)
    if (((os.label[i] * Ei < -os.toler) and (os.alpha[i] < os.c)) or
            ((os.label[i] * Ei > os.toler) and (os.alpha[i] > 0))):
        j, Ej = select_j(os, i, Ei)
        alphai_old = os.alpha[i].copy()
        alphaj_old = os.alpha[j].copy()
        if os.label[i] != os.label[j]:
            L = max(0, alphaj_old - alphai_old)
            H = min(os.c, os.c + alphaj_old - alphai_old)
        else:
            L = max(0, alphaj_old + alphai_old - os.c)
            H = min(os.c, alphaj_old + alphai_old)
        if L == H:
            print('L==H')
            return 0
        eta = -2.0 * os.k[i, j] + os.k[i, i] + os.k[j, j]
        if eta <= 0:
            print('eta<=0')
            return 0
        os.alpha[j] += os.label[j] * (Ei - Ej) / eta
        os.alpha[j] = clip_alpha(os.alpha[j], H, L)
        # update_E(os, j)
        if abs(os.alpha[j] - alphaj_old) < 0.00001:
            print('j not moving enough')
            return 0
        os.alpha[i] += os.label[i] * os.label[j] * (alphaj_old - os.alpha[j])
        # update_E(os, i)
        b1 = (os.b - Ei - os.label[i] * (os.alpha[i] - alphai_old) *
              os.k[i, i] - os.label[j] * (os.alpha[j] - alphaj_old) *
              os.k[i, j])
        b2 = (os.b - Ej - os.label[i] * (os.alpha[i] - alphai_old) *
              os.k[i, j] - os.label[j] * (os.alpha[j] - alphaj_old) *
              os.k[j, j])
        if (0 < os.alpha[i]) and (os.alpha[i] < os.c):
            os.b = b1
        elif (0 < os.alpha[j]) and (os.alpha[j] < os.c):
            os.b = b2
        else:
            os.b = (b1 + b2) / 2.0
        update_E(os, i)
        update_E(os, j)
        return 1
    else:
        return 0


def smo(data_mat_in, label_mat_in, c, toler, max_iter, ktype):
    os = data_struct(mat(data_mat_in), mat(label_mat_in).transpose(), c, toler,
                     ktype)
    iter = 0
    alpha_pairs_changed = 0
    entire_set = True
    while (iter < max_iter) and ((alpha_pairs_changed > 0) or (entire_set)):
        alpha_pairs_changed = 0
        if entire_set:
            for i in range(os.m):
                alpha_pairs_changed += innerL(os, i)
                print('fullset, iter: %d, i: %d, pairs changed: %d' %
                      (iter, i, alpha_pairs_changed))
            iter += 1
        else:
            non_bound_ids = nonzero((os.alpha.A > 0) * (os.alpha.A < c))[0]
            for i in non_bound_ids:
                alpha_pairs_changed += innerL(os, i)
                print('non-bound, iter: %d, i: %d, pairs changed: %d' %
                      (iter, i, alpha_pairs_changed))
            iter += 1
        if entire_set:
            entire_set = False
        elif alpha_pairs_changed == 0:
            entire_set = True
        print('iteration number: %d' % iter)
    return os.b, os.alpha


def calcu_w(alpha, data_arr, label_arr):
    data_mat = mat(data_arr)
    label_mat = mat(label_arr).transpose()
    m, n = shape(data_mat)
    w = zeros((n, 1))
    for i in range(m):
        w += multiply(alpha[i] * label_mat[i], data_mat[i, :].T)
    return w


def test_rbf(k1=1.3):
    data_arr, label_arr = load_data_set('testSetRBF.txt')
    b, alpha = smo(data_arr, label_arr, 200, 0.0001, 10000, ('rbf', k1))
    data_mat = mat(data_arr)
    label_mat = mat(label_arr).transpose()
    sv_id = nonzero(alpha.A > 0)[0]
    svs = data_mat[sv_id]
    sv_label = label_mat[sv_id]
    print('there are %d support vectors' % shape(svs)[0])
    m, n = shape(data_mat)
    errorcount = 0
    for i in range(m):
        kernel_vec = kernel_trans(svs, data_mat[i, :], ('rbf', k1))
        predict = kernel_vec.T * multiply(alpha[sv_id], sv_label) + b
        if sign(predict) != sign(label_arr[i]):
            errorcount += 1
    print('the training error rate is: %f' % (float(errorcount) / m))
    data_arr, label_arr = load_data_set('testSetRBF2.txt')
    data_mat = mat(data_arr)
    label_mat = mat(label_arr).transpose()
    m, n = shape(data_mat)
    errorcount = 0
    for i in range(m):
        kernel_vec = kernel_trans(svs, data_mat[i, :], ('rbf', k1))
        predict = kernel_vec.T * multiply(alpha[sv_id], sv_label) + b
        if sign(predict) != sign(label_arr[i]):
            errorcount += 1
    print('the test error rate is: %f' % (float(errorcount) / m))


def img2vector(filename):
    returnvec = zeros((1, 1024))
    with open(filename) as fr:
        for i in range(32):
            line = fr.readline()
            for j in range(32):
                returnvec[0, 32 * i + j] = int(line[j])
        return returnvec


def load_images(dirname):
    from os import listdir
    hw_labels = []
    training_list = listdir(dirname)
    m = len(training_list)
    training_mat = zeros((m, 1024))
    for i in range(m):
        file_name = training_list[i]
        file_str = file_name.split('.')[0]
        class_num = int(file_str.split('_')[0])
        if class_num == 9:
            hw_labels.append(-1)
        else:
            hw_labels.append(1)
        training_mat[i, :] = img2vector('%s/%s' % (dirname, file_name))
    return training_mat, hw_labels


def test_digits(ktype=('rbf', 10)):
    data_arr, label_arr = load_images('trainingDigits')
    b, alpha = smo(data_arr, label_arr, 200, 0.0001, 10000, ktype)
    data_mat = mat(data_arr)
    label_mat = mat(label_arr).transpose()
    sv_id = nonzero(alpha.A > 0)[0]
    svs = data_mat[sv_id]
    sv_label = label_mat[sv_id]
    print('there are %d support vectors' % shape(svs)[0])
    m, n = shape(data_mat)
    errorcount = 0
    for i in range(m):
        kernel_vec = kernel_trans(svs, data_mat[i, :], ktype)
        predict = kernel_vec.T * multiply(alpha[sv_id], sv_label) + b
        if sign(predict) != sign(label_arr[i]):
            errorcount += 1
    print('the training error rate is: %f' % (float(errorcount) / m))
    data_arr, label_arr = load_images('testDigits')
    data_mat = mat(data_arr)
    label_mat = mat(label_arr).transpose()
    m, n = shape(data_mat)
    errorcount = 0
    for i in range(m):
        kernel_vec = kernel_trans(svs, data_mat[i, :], ktype)
        predict = kernel_vec.T * multiply(alpha[sv_id], sv_label) + b
        if sign(predict) != sign(label_arr[i]):
            errorcount += 1
    print('the test error rate is: %f' % (float(errorcount) / m))

test_digits(ktype=('lin', 0))
