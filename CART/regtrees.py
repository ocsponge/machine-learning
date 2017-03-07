from numpy import *


def load_data_set(file_name):
    data_list = []
    with open(file_name) as fr:
        for line in fr.readlines():
            line_str = line.strip().split('\t')
            line_list = list(map(float, line_str))
            data_list.append(line_list)
    return data_list


def bin_split_data(data_mat, feature, value):
    mat0 = data_mat[nonzero(data_mat[:, feature] > value)[0], :]
    mat1 = data_mat[nonzero(data_mat[:, feature] <= value)[0], :]
    return mat0, mat1


def reg_leaf(data_mat):
    return mean(data_mat[:, -1])


def reg_err(data_mat):
    return var(data_mat[:, -1]) * shape(data_mat)[0]


def choose_best_split(data_mat,
                      leaf_type=reg_leaf,
                      err_type=reg_err,
                      ops=(1, 4)):
    tols = ops[0]
    toln = ops[1]
    if len(set(data_mat[:, -1].T.tolist()[0])) == 1:
        return None, leaf_type(data_mat)
    m, n = shape(data_mat)
    s = err_type(data_mat)
    bests = inf
    best_id = 0
    best_val = 0
    for i in range(n - 1):
        for val in set(data_mat[:, i].T.tolist()[0]):
            mat0, mat1 = bin_split_data(data_mat, i, val)
            if (shape(mat0)[0] < toln) or (shape(mat1)[0] < toln):
                continue
            news = err_type(mat0) + err_type(mat1)
            if news < bests:
                bests = news
                best_id = i
                best_val = val
    if (s - bests) < tols:
        return None, leaf_type(data_mat)
    mat0, mat1 = bin_split_data(data_mat, best_id, best_val)
    if (shape(mat0)[0] < toln) or (shape(mat1)[0] < toln):
        return None, leaf_type(data_mat)
    return best_id, best_val


def create_tree(data_mat, leaf_type=reg_leaf, err_type=reg_err, ops=(1, 4)):
    feat, val = choose_best_split(data_mat, leaf_type, err_type, ops)
    if feat is None:
        return val
    ret_tree = {}
    ret_tree['spid'] = feat
    ret_tree['spva'] = val
    left_mat, right_mat = bin_split_data(data_mat, feat, val)
    ret_tree['left'] = create_tree(left_mat, leaf_type, err_type, ops)
    ret_tree['right'] = create_tree(right_mat, leaf_type, err_type, ops)
    return ret_tree


def is_tree(obj):
    return isinstance(obj, dict)


def get_mean(tree):
    if is_tree(tree['left']):
        tree['left'] = get_mean(tree['left'])
    if is_tree(tree['right']):
        tree['right'] = get_mean(tree['right'])
    return (tree['left'] + tree['right']) / 2.0


def prune(tree, test_mat):
    if shape(test_mat)[0] == 0:
        return get_mean(tree)
    if is_tree(tree['left']) or is_tree(tree['right']):
        lset, rset = bin_split_data(test_mat, tree['spid'], tree['spva'])
    if is_tree(tree['left']):
        tree['left'] = prune(tree['left'], lset)
    if is_tree(tree['right']):
        tree['right'] = prune(tree['right'], rset)
    if not is_tree(tree['left']) and not is_tree(tree['right']):
        lset, rset = bin_split_data(test_mat, tree['spid'], tree['spva'])
        error_no_merge = (sum(power(lset[:, -1] - tree['left'], 2)) +
                          sum(power(rset[:, -1] - tree['right'], 2)))
        tree_mean = (tree['left'] + tree['right']) / 2.0
        error_merge = sum(power(test_mat[:, -1] - tree_mean, 2))
        if error_merge < error_no_merge:
            print('merging')
            return tree_mean
        else:
            return tree
    else:
        return tree


def linear_solve(data_mat):
    m, n = shape(data_mat)
    x = mat(ones((m, n)))
    y = mat(ones((m, 1)))
    x[:, 1:n] = data_mat[:, 0:n - 1]
    y = data_mat[:, -1]
    xTx = x.T * x
    if linalg.det(xTx) == 0:
        raise NameError('this matrix is singular, cannot do inverse')
    ws = xTx.I * x.T * y
    return ws, x, y


def model_leaf(data_mat):
    ws, x, y = linear_solve(data_mat)
    return ws


def model_err(data_mat):
    ws, x, y = linear_solve(data_mat)
    yhat = x * ws
    return sum(power(y - yhat, 2))


def reg_tree_eval(model, inmat):
    return float(model)


def model_tree_eval(model, inmat):
    n = shape(inmat)[1]
    x = mat(ones((1, n + 1)))
    x[:, 1:n + 1] = inmat
    return float(x * model)


def tree_forecast(tree, inmat, model_eval=reg_tree_eval):
    if not is_tree(tree):
        return model_eval(tree, inmat)
    if inmat[tree['spid']] > tree['spva']:
        if is_tree(tree['left']):
            return tree_forecast(tree['left'], inmat, model_eval)
        else:
            return model_eval(tree['left'], inmat)
    else:
        if is_tree(tree['right']):
            return tree_forecast(tree['right'], inmat, model_eval)
        else:
            return model_eval(tree['right'], inmat)


def create_forecast(tree, test_list, model_eval=reg_tree_eval):
    m = len(test_list)
    yhat = mat(zeros((m, 1)))
    for i in range(m):
        yhat[i, 0] = tree_forecast(tree, mat(test_list[i]), model_eval)
    return yhat


if __name__ == '__main__':
    trainmat = mat(load_data_set('bikeSpeedVsIq_train.txt'))
    testmat = mat(load_data_set('bikeSpeedVsIq_test.txt'))
    tree = create_tree(trainmat, model_leaf, model_err, ops=(1, 20))
    yhat = create_forecast(tree, testmat[:, 0], model_tree_eval)
    cc = corrcoef(yhat, testmat[:, 1], rowvar=0)[0, 1]
    print(cc)
    ws, x, y = linear_solve(trainmat)
    for i in range(shape(testmat)[0]):
        yhat[i] = testmat[i, 0] * ws[1, 0] + ws[0, 0]
    cc = corrcoef(yhat, testmat[:, 1], rowvar=0)[0, 1]
    print(cc)
