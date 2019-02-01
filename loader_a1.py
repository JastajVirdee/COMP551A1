import json
from collections import Counter
import numpy as np

def text_cleaner(string):
    """
    returns a list of words, defining word as separated on white space.
    strips out empty strs from the list
    """
    string = string.lower().replace('\n',' ').split(" ")

    return [x for x in string if x!='']

def get_freq(data_set):
    """
    Returns a list of the top 160 scored words
    use collections.counter to determine.
    """
    count = Counter()

    for d in data_set:
        count.update(d['text'])

    top_160 = count.most_common(160)
    top_words = [x for (x,_) in top_160]

    return top_words

def build_vector(comment_text, words):
    """
    takes in the text list and the top 160 words
    checks all words in the comment's text to see if they're
    in the top 160 words, if yes then add them to the
    feature vector at index = rank of the popular word

    return that feature vector.
    """
    vec = [0]*160

    for w in comment_text:
        if w in words:
           i = words.index(w)
           vec[i] += 1

    return vec

def add_vectors(data_set, words):
    """
    add to the dictionary of each data point the feature vector
    that describes which top 160 popular words it has in it.
    """
    for dp in data_set:
        v = build_vector(dp['text'], words)
        dp['xcounts'] = v

def build_vector_60(comment_text, words):
    """
    takes in the text list and the top 160 words
    checks all words in the comment's text to see if they're
    in the top 60 words, if yes then add them to the
    feature vector at index = rank of the popular word

    return that feature vector.
    """
    vec = [0]*60

    for w in comment_text:
        if w in words:
           i = words.index(w)
           vec[i] += 1

    return vec

def add_vectors_60(data_set, words):
    """
    add to the dictionary of each data point the feature vector
    that describes which top 60 popular words it has in it.
    """
    for dp in data_set:
        v = build_vector_60(dp['text'], words)
        dp['6counts'] = v

def make_matrix(data_set):
    """
    turns each dictionary in a data block into a row of a numpy matrix
    including the new features

    Call on each data partition.
    Returns matrix X containing all features for data pts,
            array y containing all target scores.
    """
    X = []
    y = []

    for d in data_set:
        X.append([1, d['children'], d['controversiality'], int(d['is_root']),d['number_words'], d['offensive'] ] + d['xcounts'])
        y.append(d['popularity_score'])

    return (np.array(X), np.array(y))

def make_matrix_3_1(data_set):
    """
    Same as previous matrix generator, but without the popularity counts feature
    or the homebrew features
    """

    X = []
    y = []

    for d in data_set:
        X.append([1, d['children'], d['controversiality'], int(d['is_root'])])
        y.append(d['popularity_score'])

    return (np.array(X), np.array(y))

def make_matrix_60(data_set):
    """
    Generates matrix, target vector with top 60 words instead of 160
     + no new features
    """
    X = []
    y = []
    for d in data_set:
        X.append([1, d['children'], d['controversiality'], int(d['is_root'])] + d['6counts'])
        y.append(d['popularity_score'])
    return (np.array(X), np.array(y))

def make_matrix_160(data_set):
    """
    Generates matrix with all 160 words and basic fetaures but no bonus features.
    """
    X = []
    y = []
    for d in data_set:
        X.append([1, d['children'], d['controversiality'], int(d['is_root']) ] + d['xcounts'])
        y.append(d['popularity_score'])
    return (np.array(X), np.array(y))


def make_matrix_60_newfeatures(data_set):
    """
    60 words + new features matrix and target scores vector.
    """
    X = []
    y = []

    for d in data_set:
        X.append([1, d['children'], d['controversiality'], int(d['is_root']),d['number_words'], d['offensive'] ] + d['6counts'])
        y.append(d['popularity_score'])

    return (np.array(X), np.array(y))

def load_data():
    """
    The main loader that prepares dictionaries with all new features
    Note: make_matrix should be called on the returned sets

    returns partitions of the data (train val test) with all additional features added.
    """

    with open("proj1_data.json") as fp:
        data = json.load(fp)

    with open("offensive_words.json") as fp:
        offensive_words = json.load(fp)


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

    # partition initial data points
    train_set = data[0:10000]
    val_set = data[10000:11000]
    test_set = data[11000:12000]

    # get the most frequent words from the training data only.
    words = get_freq(train_set)

    # update the dictionary objects with top 160 words feature vectors
    add_vectors(train_set, words)
    add_vectors(test_set, words)
    add_vectors(val_set, words)

    # update dictionary objects with top 60 words feature vectors.
    add_vectors_60(train_set, words[0:60])
    add_vectors_60(test_set, words[0:60])
    add_vectors_60(val_set, words[0:60])

    return (train_set, val_set, test_set)
