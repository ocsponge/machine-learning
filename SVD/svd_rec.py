from numpy import *
from numpy import linalg as la


def load_data():
    return [[4, 4, 0, 2, 2], [4, 0, 0, 3, 3], [4, 0, 0, 1, 1], [1, 1, 1, 2, 0],
            [2, 2, 2, 0, 0], [1, 1, 1, 0, 0], [5, 5, 5, 0, 0]]


def load_data2():
    return [
        [0, 0, 0, 0, 0, 4, 0, 0, 0, 0, 5], [0, 0, 0, 3, 0, 4, 0, 0, 0, 0, 3],
        [0, 0, 0, 0, 4, 0, 0, 1, 0, 4, 0], [3, 3, 4, 0, 0, 0, 0, 2, 2, 0, 0],
        [5, 4, 5, 0, 0, 0, 0, 5, 5, 0, 0], [0, 0, 0, 0, 5, 0, 1, 0, 0, 5, 0],
        [4, 3, 4, 0, 0, 0, 0, 5, 5, 0, 1], [0, 0, 0, 4, 0, 4, 0, 0, 0, 0, 4],
        [0, 0, 0, 2, 0, 2, 5, 0, 0, 1, 2], [0, 0, 0, 0, 5, 0, 0, 0, 0, 4, 0],
        [1, 0, 0, 0, 0, 0, 0, 1, 2, 0, 0]
    ]


def eu_sim(ina, inb):
    return 1.0 / (1.0 + la.norm(ina - inb))


def pe_sim(ina, inb):
    if len(ina) < 3:
        return 1.0
    return 0.5 + 0.5 * corrcoef(ina, inb, rowvar=0)[0][1]


def cos_sim(ina, inb):
    num = float(ina.T * inb)
    denom = la.norm(ina) * la.norm(inb)
    return 0.5 + 0.5 * (num / denom)


def calc_score(data_mat, user, item, sim_meas):
    n = shape(data_mat)[1]
    sim_total = 0.0
    rat_total = 0.0
    for j in range(n):
        user_rate = data_mat[user, j]
        if user_rate == 0:
            continue
        common = nonzero(
            logical_and(data_mat[:, item] > 0, data_mat[:, j] > 0))[0]
        if len(common) == 0:
            sim = 0.0
        else:
            sim = sim_meas(data_mat[common, j], data_mat[common, item])
        print('the %d and %d similarity is: %f' % (item, j, sim))
        sim_total += sim
        rat_total += sim * user_rate
    if sim_total == 0:
        return 0
    else:
        return rat_total / sim_total


def svd_score(data_mat, user, item, sim_meas):
    n = shape(data_mat)[1]
    sim_total = 0.0
    rat_total = 0.0
    u, sigma, vt = la.svd(data_mat)
    sig_mat = mat(eye(4) * sigma[:4])
    data_trans = data_mat.T * u[:, :4] * sig_mat.I
    for j in range(n):
        user_rate = data_mat[user, j]
        if user_rate == 0 or j == item:
            continue
        sim = sim_meas(data_trans[item, :].T, data_trans[j, :].T)
        print('the %d and %d similarity is: %f' % (item, j, sim))
        sim_total += sim
        rat_total += sim * user_rate
    if sim_total == 0:
        return 0
    else:
        return rat_total / sim_total


def recommend(data_mat, user, N=3, sim_meas=cos_sim, est_method=calc_score):
    unrated = nonzero(data_mat[user, :] == 0)[1]
    if len(unrated) == 0:
        return 'you rated everything'
    item_scores = []
    for item in unrated:
        score = est_method(data_mat, user, item, sim_meas)
        item_scores.append((item, score))
    return sorted(item_scores, key=lambda jj: jj[1], reverse=True)[:N]


def print_mat(inmat, thresh=0.8):
    for i in range(32):
        for j in range(32):
            if float(inmat[i, j]) > thresh:
                print(1, end='')
            else:
                print(0, end='')
        print('')


def img_comp(numsv=3, thresh=0.8):
    text = []
    with open('0_5.txt') as fr:
        for line in fr.readlines():
            row = []
            for i in range(32):
                row.append(int(line[i]))
            text.append(row)
    print('*****origin matrix*****')
    text_mat = mat(text)
    print_mat(text_mat, thresh)
    u, sigma, vt = la.svd(text_mat)
    sig = mat(eye(numsv) * sigma[:numsv])
    text_trans = u[:, :numsv] * sig * vt[:numsv, :]
    print('*****reconstructed matrix using %d singular values*****' % numsv)
    print_mat(text_trans, thresh)


img_comp(2)
