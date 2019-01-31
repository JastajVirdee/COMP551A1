import json
from collections import Counter
import numpy as np

def text_cleaner(string):
    string = string.lower().replace('\n',' ').split(" ")
    return [x for x in string if x!='']

def get_freq(data_set):
    count = Counter()
    for d in data_set:
        count.update(d['text'])
    top_160 = count.most_common(160)
    #print(top_160)
    top_words = [x for (x,_) in top_160]
    return top_words

def build_vector(comment_text, words):
    vec = [0]*160
    for w in comment_text:
        if w in words:
           i = words.index(w)
           vec[i] += 1
    return vec

def add_vectors(data_set, words):
    for dp in data_set:
        v = build_vector(dp['text'], words)
        dp['xcounts'] = v

def build_vector_60(comment_text, words):
    vec = [0]*60
    for w in comment_text:
        if w in words:
           i = words.index(w)
           vec[i] += 1
    return vec

def add_vectors_60(data_set, words):
    for dp in data_set:
        v = build_vector_60(dp['text'], words)
        dp['6counts'] = v

def make_matrix(data_set):
    X = []
    y = []
    for d in data_set:
        X.append([1, d['children'], d['controversiality'], int(d['is_root']),d['number_words'], d['offensive'] ] + d['xcounts'])
        y.append(d['popularity_score'])
    return (np.array(X), np.array(y))

def make_matrix_3_1(data_set):
    X = []
    y = []
    for d in data_set:
        X.append([1, d['children'], d['controversiality'], int(d['is_root'])])
        y.append(d['popularity_score'])
    return (np.array(X), np.array(y))

def make_matrix_60(data_set):
    X = []
    y = []
    for d in data_set:
        X.append([1, d['children'], d['controversiality'], int(d['is_root'])] + d['6counts'])
        y.append(d['popularity_score'])
    return (np.array(X), np.array(y))

def make_matrix_160(data_set):
    X = []
    y = []
    for d in data_set:
        X.append([1, d['children'], d['controversiality'], int(d['is_root']) ] + d['xcounts'])
        y.append(d['popularity_score'])
    return (np.array(X), np.array(y))


def make_matrix_60_newfeatures(data_set):
    X = []
    y = []
    for d in data_set:
        X.append([1, d['children'], d['controversiality'], int(d['is_root']),d['number_words'], d['offensive'] ] + d['6counts'])
        y.append(d['popularity_score'])
    return (np.array(X), np.array(y))

def load_data():

    with open("proj1_data.json") as fp:
        data = json.load(fp)

    with open("offensive_words.json") as fp:
        offensive_words = json.load(fp)

    #print(offensive_words)

    for d in data:
        d['text'] = text_cleaner(d['text'])
        #d['number_words'] = len(d['text'])
        if (len(d['text']) > 5):
            d['number_words'] = 1
        else:
            d['number_words'] = 0
        d['offensive'] = 0
        for w in offensive_words:
            if w in d['text']:
                d['offensive'] = 1
                break

    train_set = data[0:10000]
    val_set = data[10000:11000]
    test_set = data[11000:12000]

    words = get_freq(train_set)
    #with open('words.txt', 'w') as wfp:
    #    w_string = ' '.join(words)
    #    wfp.write(w_string)
    #print(words)

    add_vectors(train_set, words)
    add_vectors(test_set, words)
    add_vectors(val_set, words)

    add_vectors_60(train_set, words[0:60])
    add_vectors_60(test_set, words[0:60])
    add_vectors_60(val_set, words[0:60])

    #print(train_set[0:5])

    return (train_set, val_set, test_set)

#X, y = make_matrix(train_set)

#train, val, test = load_data()
#X_train, y_train = make_matrix(train)
#X_val, y_val = make_matrix(val)
#X_test, y_test = make_matrix(val)

#print(X_train)

#print(np.matmul(X,X.transpose()))
#print(y)
