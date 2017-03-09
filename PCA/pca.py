from numpy import *
import matplotlib.pyplot as plt


def load_data_mat(filename, delim='\t'):
    with open(filename) as fr:
        str_list = [line.strip().split(delim) for line in fr.readlines()]
        data_list = [list(map(float, one_str)) for one_str in str_list]
    return mat(data_list)


def pca(data_mat, topN_feat=9999999):
    mean_vec = mean(data_mat, 0)
    mean_removed = data_mat - mean_vec
    covmat = cov(mean_removed, rowvar=0)
    eig_vals, eig_vecs = linalg.eig(mat(covmat))
    eig_ids = argsort(eig_vals)
    eig_ids = eig_ids[:-(topN_feat + 1):-1]
    last_vecs = eig_vecs[:, eig_ids]
    low_mat = mean_removed * last_vecs
    low_data = low_mat * last_vecs.T + mean_vec
    return low_mat, low_data


def replace_nan():
    data_mat = load_data_mat('secom.data', ' ')
    num_col = shape(data_mat)[1]
    for j in range(num_col):
        mean_val = mean(data_mat[nonzero(~isnan(data_mat[:, j]))[0], j])
        data_mat[nonzero(isnan(data_mat[:, j]))[0], j] = mean_val
    return data_mat


'''
data_mat = load_data_mat('testSet.txt')
lowmat, lowdat = pca(data_mat, 1)
fig = plt.figure()
ax = fig.add_subplot(111)
ax.scatter(data_mat[:, 0], data_mat[:, 1], marker='^', s=90)
ax.scatter(lowdat[:, 0], lowdat[:, 1], marker='o', s=50, c='red')
plt.show()
'''
data_mat = replace_nan()
mean_vec = mean(data_mat, 0)
mean_removed = data_mat - mean_vec
covmat = cov(mean_removed, rowvar=0)
eig_vals, eig_vecs = linalg.eig(mat(covmat))
per = [val / sum(eig_vals) for val in eig_vals]
print(per[19])
