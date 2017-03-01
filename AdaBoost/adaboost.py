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
        # print('D: ', D.T)
        alpha = float(0.5 * log((1 - error) / max(error, 1e-16)))
        stump['alpha'] = alpha
        weak_class_arr.append(stump)
        # print('classify: ', classify.T)
        expo = multiply(-1 * alpha * mat(class_label).T, classify)
        D = multiply(exp(expo), D)
        D = D / D.sum()
        weight_classify += alpha * classify
        # print('weight_classify: ', weight_classify.T)
        weight_error = multiply(sign(weight_classify) != mat(class_label).T,
                                ones((m, 1)))
        error_rate = weight_error.sum() / m
        print('total error: ', error_rate, '\n')
        if error_rate == 0:
            break
    return weak_class_arr, weight_classify


def ada_classify(data_arr, classify_list):
    data_mat = mat(data_arr)
    m = shape(data_mat)[0]
    result = mat(zeros((m, 1)))
    for i in range(len(classify_list)):
        one_classify = stump_classify(
            data_mat, classify_list[i]['dim'],
            classify_list[i]['thresh'], classify_list[i]['ineq'])
        result += classify_list[i]['alpha'] * one_classify
        print(result)
    return sign(result)


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


def plot_roc(predict, class_label):
    import matplotlib.pyplot as plt
    cur = (1.0, 1.0)
    ysum = 0.0
    num_positive = sum(array(class_label) == 1.0)
    xstep = 1.0 / float(len(class_label) - num_positive)
    ystep = 1.0 / float(num_positive)
    sorted_id = predict.argsort()
    fig = plt.figure()
    fig.clf()
    ax = plt.subplot(111)
    for i in sorted_id.tolist()[0]:
        if class_label[i] == 1.0:
            delx = 0
            dely = ystep
        else:
            delx = xstep
            dely = 0
            ysum += cur[1]
        ax.plot([cur[0], cur[0] - delx], [cur[1], cur[1] - dely], c='b')
        cur = (cur[0] - delx, cur[1] - dely)
    ax.plot([0, 1], [0, 1], 'b--')
    plt.xlabel('false positive rate')
    plt.ylabel('true positive rate')
    plt.title('roc curve for adaboost horse colic detection system')
    ax.axis([0, 1, 0, 1])
    plt.show()
    print('the area under curve is: ', ysum * xstep)

data_arr, label_arr = load_data_set('horseColicTraining2.txt')
classarr, classify = adaboost_train_ds(data_arr, label_arr, 50)
plot_roc(classify.T, label_arr)
'''testarr, testlabel = load_data_set('horseColicTest2.txt')
predict = ada_classify(testarr, classarr)
errarr = mat(ones((67, 1)))
numerr = errarr[predict != mat(testlabel).T].sum()
errorrate = float(numerr) / 67.0
print(numerr)
print(errorrate)'''
