class TreeNode:

    def __init__(self, name, num, parent):
        self.name = name
        self.num = num
        self.parent = parent
        self.link = None
        self.children = {}

    def increase(self, num):
        self.num += num

    def display(self, id=1):
        print('  ' * id, self.name, ' ', self.num)
        for child in self.children.values():
            child.display(id + 1)


def load_simp_dat():
    simpDat = [['r', 'z', 'h', 'j', 'p'],
               ['z', 'y', 'x', 'w', 'v', 'u', 't', 's'],
               ['z'],
               ['r', 'x', 'n', 'o', 's'],
               ['y', 'r', 'x', 'z', 'q', 't', 'p'],
               ['y', 'z', 'x', 'e', 'q', 's', 't', 'm']]
    return simpDat


def create_set(data_list):
    retdict = {}
    for item in data_list:
        retdict[frozenset(item)] = 1
    return retdict


def update_link(start_node, end_node):
    while (start_node.link is not None):
        start_node = start_node.link
    start_node.link = end_node


def update_tree(item, intree, header_dict, num):
    if item[0] in intree.children:
        intree.children[item[0]].increase(num)
    else:
        intree.children[item[0]] = TreeNode(item[0], num, intree)
        if header_dict[item[0]][1] is None:
            header_dict[item[0]][1] = intree.children[item[0]]
        else:
            update_link(header_dict[item[0]][1], intree.children[item[0]])
    if len(item) > 1:
        update_tree(item[1::], intree.children[item[0]], header_dict, num)


def create_tree(data_dict, min=1):
    header_dict = {}
    for trans in data_dict:
        for item in trans:
            header_dict[item] = header_dict.get(item, 0) + data_dict[trans]
    for item in list(header_dict):
        if header_dict[item] < min:
            del (header_dict[item])
    freqitems = set(header_dict.keys())
    if len(freqitems) == 0:
        return None, None
    for item in header_dict:
        header_dict[item] = [header_dict[item], None]
    rettree = TreeNode('NullSet', 1, None)
    for (trans, num) in data_dict.items():
        local = {}
        for item in trans:
            if item in freqitems:
                local[item] = header_dict[item][0]
        if len(local) > 0:
            order_tran = [v[0] for v in sorted(
                local.items(), key=lambda p: p[1], reverse=True)]
            # print(order_tran)
            order_tran = sorted(order_tran, key=lambda l:
                                (header_dict[l][0], l), reverse=True)
            # print(order_tran)
            update_tree(order_tran, rettree, header_dict, num)
    return rettree, header_dict


def ascend_tree(node, path):
    if node.parent is not None:
        path.append(node.name)
        ascend_tree(node.parent, path)


def find_path(base_elem, node):
    related_elem = {}
    while(node is not None):
        path = []
        ascend_tree(node, path)
        if len(path) > 1:
            related_elem[frozenset(path[1:])] = node.num
        node = node.link
    return related_elem


def mine_tree(head_dict, min, related_elem, freq_lists):
    items = [v[0] for v in sorted(head_dict.items(), key=lambda p:p[1][0])]
    for it in items:
        new_list = related_elem.copy()
        new_list.add(it)
        freq_lists.append(new_list)
        new_related = find_path(it, head_dict[it][1])
        new_tree, new_head = create_tree(new_related, min)
        if new_head is not None:
            # print('conditional tree for: ', new_list)
            # new_tree.display()
            mine_tree(new_head, min, new_list, freq_lists)


simdat = [line.split() for line in open('kosarak.dat').readlines()]
simset = create_set(simdat)
tree, head = create_tree(simset, 100000)
x = []
mine_tree(head, 100000, set([]), x)
print(x)
