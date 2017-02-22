from numpy import *


def load_data_set():
    posting_list = [['my', 'dog', 'has', 'flea', 'problems', 'help', 'please'],
                    ['maybe', 'not', 'take', 'him', 'to', 'dog',
                     'park', 'stupid'],
                    ['my', 'dalmation', 'is', 'so', 'cute', 'I', 'love',
                     'him'],
                    ['stop', 'posting', 'stupid', 'worthless', 'garbage'],
                    ['mr', 'licks', 'ate', 'my', 'steak', 'how', 'to', 'stop',
                     'him'],
                    ['quit', 'buying', 'worthless', 'dog', 'food', 'stupid']]
    class_vec = [0, 1, 0, 1, 0, 1]
    return posting_list, class_vec


def create_vocab_list(data_set):
    vocab_set = set([])
    for document in data_set:
        vocab_set = vocab_set | set(document)
    return list(vocab_set)


def set_words2vec(vocab_list, input_set):
    return_vec = [0] * len(vocab_list)
    for word in input_set:
        if word in vocab_list:
            return_vec[vocab_list.index(word)] = 1
        else:
            print('the word: %s is not in my vocabulary!' % word)
    return return_vec


def train_nb0(train_matrix, train_category):
    num_train_docs = len(train_matrix)
    num_words = len(train_matrix[0])
    p_abusive = sum(train_category) / float(num_train_docs)
    p0_num = ones(num_words)
    p1_num = ones(num_words)
    p0_denom = 2.0
    p1_denom = 2.0
    for i in range(num_train_docs):
        if (train_category[i] == 1):
            p1_num += train_matrix[i]
            p1_denom += 1
        else:
            p0_num += train_matrix[i]
            p0_denom += 1
    p1_vect = log(p1_num / p1_denom)
    p0_vect = log(p0_num / p0_denom)
    return p0_vect, p1_vect, p_abusive


def classify_nb(vec_classify, p0_vec, p1_vec, p_class1):
    p1 = sum(vec_classify * p1_vec) + log(p_class1)
    p0 = sum(vec_classify * p0_vec) + log(1 - p_class1)
    if p1 > p0:
        return 1
    else:
        return 0


def testing():
    list_post, list_class = load_data_set()
    my_vocab = create_vocab_list(list_post)
    train = []
    for doc in list_post:
        train.append(set_words2vec(my_vocab, doc))
    p0v, p1v, pab = train_nb0(train, list_class)
    test_entry = ['love', 'my', 'dalmation']
    this_doc = set_words2vec(my_vocab, test_entry)
    print(test_entry, 'classified as: ', classify_nb(this_doc, p0v, p1v, pab))
    test_entry = ['stupid', 'garbage']
    this_doc = set_words2vec(my_vocab, test_entry)
    print(test_entry, 'classified as: ', classify_nb(this_doc, p0v, p1v, pab))


def bag_words2vec(vocab_list, input_set):
    return_vec = [0] * len(vocab_list)
    for word in input_set:
        if word in vocab_list:
            return_vec[vocab_list.index(word)] += 1
    return return_vec


def text_parse(big_str):
    import re
    list_token = re.split(r'[^a-zA-Z0-9_]', big_str)
    return [tok.lower() for tok in list_token if len(tok) > 2]


def spam_test():
    doc_list = []
    class_list = []
    for i in range(1, 26):
        with open('email/spam/%d.txt' % i, 'rb') as f1:
            word_list = text_parse(f1.read().decode('gbk'))
            doc_list.append(word_list)
            class_list.append(1)
    for i in range(1, 26):
        with open('email/ham/%d.txt' % i, 'rb') as f2:
            word_list = text_parse(f2.read().decode('gbk', errors='ignore'))
            doc_list.append(word_list)
            class_list.append(0)
    vocab_list = create_vocab_list(doc_list)
    training_id = list(range(50))
    test_set = []
    training_set = []
    training_class = []
    for i in range(10):
        ran_index = int(random.uniform(0, len(training_id)))
        test_set.append(training_id[ran_index])
        del(training_id[ran_index])
    for id in training_id:
        training_set.append(set_words2vec(vocab_list, doc_list[id]))
        training_class.append(class_list[id])
    p0v, p1v, pab = train_nb0(training_set, training_class)
    errorcount = 0.0
    for index in test_set:
        test_input = set_words2vec(vocab_list, doc_list[index])
        if classify_nb(test_input, p0v, p1v, pab) != class_list[index]:
            errorcount += 1
    print('the erorr rate is: %.2f' % (float(errorcount) / len(test_set)))

spam_test()
