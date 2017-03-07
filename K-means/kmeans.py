from numpy import *
import matplotlib.pyplot as plt


def load_data_set(file_name):
    data_list = []
    with open(file_name) as fr:
        for line in fr.readlines():
            line_str = line.strip().split('\t')
            line_list = list(map(float, line_str))
            data_list.append(line_list)
    return data_list


def dist_eclud(veca, vecb):
    return sqrt(sum(power(veca - vecb, 2)))


def rand_cent(data_mat, k):
    n = shape(data_mat)[1]
    cent_ids = mat(zeros((k, n)))
    for j in range(n):
        minj = min(data_mat[:, j])
        rangej = float(max(data_mat[:, j]) - minj)
        cent_ids[:, j] = minj + rangej * random.rand(k, 1)
    return cent_ids


def kmeans(data_mat, k, dist_meas=dist_eclud, create_cent=rand_cent):
    m = shape(data_mat)[0]
    cluster_assment = mat(zeros((m, 2)))
    cent_ids = create_cent(data_mat, k)
    cluster_changed = True
    while cluster_changed:
        cluster_changed = False
        for i in range(m):
            minds = inf
            minid = -1
            for j in range(k):
                dsij = dist_meas(data_mat[i, :], cent_ids[j, :])
                if dsij < minds:
                    minds = dsij
                    minid = j
            if cluster_assment[i, 0] != minid:
                cluster_changed = True
            cluster_assment[i, :] = minid, minds**2
        # print(cent_ids)
        for cent in range(k):
            pts_cluster = data_mat[nonzero(cluster_assment[:, 0] == cent)[0]]
            cent_ids[cent, :] = mean(pts_cluster, 0)
    return cent_ids, cluster_assment


def bi_kmeans(data_mat, k, dist_meas=dist_eclud):
    m = shape(data_mat)[0]
    cluster_assment = mat(zeros((m, 2)))
    cent_id0 = mean(data_mat, 0).tolist()[0]
    cent_list = [cent_id0]
    for i in range(m):
        cluster_assment[i, 1] = dist_meas(data_mat[i, :], mat(cent_id0))
    while len(cent_list) < k:
        min_sse = inf
        best_id = -1
        for j in range(len(cent_list)):
            pts_clust = data_mat[nonzero(cluster_assment[:, 0] == j)[0]]
            centj, clustj = kmeans(pts_clust, 2, dist_meas)
            sse_split = sum(clustj[:, 1])
            sse_nosplit = sum(
                cluster_assment[nonzero(cluster_assment[:, 0] != j)[0], 1])
            if (sse_split + sse_nosplit) < min_sse:
                min_sse = sse_split + sse_nosplit
                best_id = j
                best_cent = centj
                best_clust = clustj
        best_clust[nonzero(best_clust[:, 0] == 1)[0], 0] = len(cent_list)
        best_clust[nonzero(best_clust[:, 0] == 0)[0], 0] = best_id
        print('the best cent to split is: ', best_id)
        print('the len of best clust is: ', len(best_clust))
        cent_list[best_id] = best_cent[0].tolist()[0]
        cent_list.append(best_cent[1].tolist()[0])
        cluster_assment[nonzero(cluster_assment[:, 0] == best_id)[
            0]] = best_clust
    return cent_list, cluster_assment


data_mat = mat(load_data_set('testSet2.txt'))
fig = plt.figure()
ax = fig.add_subplot(111)
ax.scatter(data_mat[:, 0], data_mat[:, 1], c='grey')
cent, cluster = bi_kmeans(data_mat, 3)
cent = mat(cent)
ax.scatter(cent[:, 0], cent[:, 1], s=100, marker='^')
plt.show()
