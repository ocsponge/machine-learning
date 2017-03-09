def load_data_list():
    return [[1, 3, 4], [2, 3, 5], [1, 2, 3, 5], [2, 5]]


def create_c1(data_list):
    c1 = []
    for i in data_list:
        for item in i:
            if not [item] in c1:
                c1.append([item])
    c1.sort()
    return list(map(frozenset, c1))


def scan(D, ck, min_support):
    num_item = {}
    for item in ck:
        for data in D:
            if item.issubset(data):
                if item not in num_item:
                    num_item[item] = 1
                else:
                    num_item[item] += 1
    total_num = float(len(D))
    support = {}
    retlist = []
    for key in num_item:
        sup_item = num_item[key] / total_num
        if sup_item >= min_support:
            retlist.insert(0, key)
        support[key] = sup_item
    return retlist, support


def create_ck(lk, k):
    lenlk = len(lk)
    retlist = []
    for i in range(lenlk):
        for j in range(i + 1, lenlk):
            l1 = list(lk[i])[:k - 2]
            l2 = list(lk[j])[:k - 2]
            l1.sort()
            l2.sort()
            if l1 == l2:
                retlist.append(lk[i] | lk[j])
    return retlist


def apriori(data_list, min_support=0.5):
    c1 = create_c1(data_list)
    D = list(map(set, data_list))
    l1, support = scan(D, c1, min_support)
    k = 2
    l = [l1]
    while (len(l[k - 2]) > 0):
        ck = create_ck(l[k - 2], k)
        lk, su = scan(D, ck, min_support)
        l.append(lk)
        support.update(su)
        k += 1
    return l, support


def rules_from_freqset(freqset, H, support, rulelist, minconf=0.7):
    m = len(H[0])
    if (len(freqset) > (m + 1)):
        Hmp1 = create_ck(H, m + 1)
        Hmp1 = calc_conf(freqset, Hmp1, support, rulelist, minconf)
        if (len(Hmp1) > 1):
            rules_from_freqset(freqset, Hmp1, support, rulelist, minconf)


def calc_conf(freqset, H, support, rulelist, minconf=0.7):
    newH = []
    for after in H:
        conf = support[freqset] / support[freqset - after]
        if conf >= minconf:
            print(freqset - after, '-->', after, 'conf: ', conf)
            rulelist.append((freqset - after, after, conf))
            newH.append(after)
    return newH


def create_rules(l, support, minconf=0.7):
    rulelist = []
    for i in range(1, len(l)):
        for freqset in l[i]:
            H1 = [frozenset([item]) for item in freqset]
            if (i > 1):
                rules_from_freqset(freqset, H1, support, rulelist, minconf)
            else:
                calc_conf(freqset, H1, support, rulelist, minconf)
    return rulelist


mushroom_set = [line.split() for line in open('mushroom.dat').readlines()]
l, su = apriori(mushroom_set, 0.3)
for item in l[3]:
    if item.intersection('2'):
        print(item)
