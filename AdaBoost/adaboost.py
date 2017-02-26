from numpy import *


def load_simp_data():
    data_mat = matrix(
        [[1.0, 2.1], [2.0, 1.1], [1.3, 1.0], [1.0, 1.0], [2.0, 1.0]])
    class_label = [1.0, 1.0, -1.0, -1.0, 1.0]
    return data_mat, class_label


def stump_classify(data_mat, dimen, thresh_val, thresh_ineq):
    result_classify = ones((shape(data_mat)[0], 1))
    if thresh_ineq == 'lt':
        result_classify[data_mat[:, dimen] <= thresh_val] = -1.0
    else:
        result_classify[data_mat[:, dimen] > thresh_val] = -1.0
    return result_classify


def bulid_stump(data_arr, label_arr, D):
    data_mat = mat(data_arr)
    label_mat = mat(label_arr).T
    m, n = shape(data_mat)
    num_step = 10.0
    best_stump = {}
    min_error = inf
    best_classify = mat(zeros((m, 1)))
    for i in range(n):
        range_min = data_mat[:, i].min()
        range_max = data_mat[:, i].max()
        step_size = (range_max - range_min) / num_step
        for j in range(-1, int(num_step) + 2):
            for ineq in ['lt', 'gt']:
                thresh_val = range_min + float(j) * step_size
                predict = stump_classify(data_mat, i, thresh_val, ineq)
                error = mat(ones((m, 1)))
                error[predict == label_mat] = 0
                weight_error = D.T * error
                '''print("split: dim % d, thresh % .2f, thresh ineqal: % s, \
the weighted error is % .3f" %
                      (i, thresh_val, ineq, weight_error))'''
                if weight_error < min_error:
                    min_error = weight_error
                    best_classify = predict.copy()
                    best_stump['dim'] = i
                    best_stump['thresh'] = thresh_val
                    best_stump['ineq'] = ineq
    return best_stump, min_error, best_classify


def adaboost_train_ds(data_arr, class_label, num_iter=40):
    weak_class_arr = []
    m = shape(data_arr)[0]
    D = mat(ones((m, 1)) / m)
    weight_classify = mat(zeros((m, 1)))
    for iter in range(num_iter):
        stump, error, classify = bulid_stump(data_arr, class_label, D)
        print('D: ', D.T)
        alpha = float(0.5 * log((1 - error) / max(error, 1e-16)))
        stump['alpha'] = alpha
        weak_class_arr.append(stump)
        print('classify: ', classify.T)
        expo = multiply(-1 * alpha * mat(class_label).T, classify)
        D = multiply(exp(expo), D)
        D = D / D.sum()
        weight_classify += alpha * classify
        print('weight_classify: ', weight_classify.T)
        weight_error = multiply(sign(weight_classify) != mat(class_label).T,
                                ones((m, 1)))
        error_rate = weight_error.sum() / m
        print('total error: ', error_rate, '\n')
        if error_rate == 0:
            break
    return weak_class_arr

data_mat, label_mat = load_simp_data()
classarr = adaboost_train_ds(data_mat, label_mat, 9)
