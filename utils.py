from keras.preprocessing import sequence
from keras.preprocessing.text import Tokenizer
import pandas as pd
import numpy as np
import dill


def get_tokenizer(tokenizer_path='../we/tokenizer.pkl'):
    """
        Load Tokenizer
    """
    # Make Tokenizer (load or from dataset)
    from keras.preprocessing.text import Tokenizer
    tokenizer = Tokenizer()

    with open(tokenizer_path, 'rb') as fi:
        tokenizer = dill.load(fi)
    return tokenizer

def get_ce_dataset(tokenizer_path='../we/tokenizer.pkl'):
    tokenizer = get_tokenizer(tokenizer_path)
    
    """
        Construct X and y
    """
    df = pd.read_csv("data/train_data.csv", delimiter=";", header=0, encoding = "ISO-8859-1")
    df_test = pd.read_csv("data/test_data.csv", delimiter=";", header=0, encoding = "ISO-8859-1")

    df = df.sample(frac=1, random_state=7)

    X = df['review']
    X_test = df_test['review']

    X = tokenizer.texts_to_sequences(X)
    X_test = tokenizer.texts_to_sequences(X_test)

    max_review_length = 150
    PADDING_TYPE = 'post'
    X = sequence.pad_sequences(X, maxlen=max_review_length, padding=PADDING_TYPE)
    X_test = sequence.pad_sequences(X_test, maxlen=max_review_length, padding=PADDING_TYPE)

    y = df[['food', 'service', 'price', 'place']]
    y = y.replace(to_replace='yes', value=1)
    y = y.replace(to_replace='no', value=0)
    y = y.replace(to_replace=np.nan, value=0)

    y_test = df_test[['food', 'service', 'price', 'place']]
    y_test = y_test.replace(to_replace='yes', value=1)
    y_test = y_test.replace(to_replace='no', value=0)
    y_test = y_test.replace(to_replace=np.nan, value=0)

    return X, y, X_test, y_test

def get_spc_dataset(category, tokenizer_path='../we/tokenizer.pkl'):
    tokenizer = get_tokenizer(tokenizer_path)

    """
        Construct X and y
    """
    df = pd.read_csv("data/train_data_3.csv", delimiter=";", header=0, encoding = "ISO-8859-1")
    df_test = pd.read_csv("data/test_data_3.csv", delimiter=";", header=0, encoding = "ISO-8859-1")

    df = df.sample(frac=1, random_state=7)

    X = df[df[category] != '-' ]['review']
    X_test = df_test[df_test[category] != '-' ]['review']

    X = tokenizer.texts_to_sequences(X)
    X_test = tokenizer.texts_to_sequences(X_test)

    max_review_length = 150
    PADDING_TYPE = 'post'
    X = sequence.pad_sequences(X, maxlen=max_review_length, padding=PADDING_TYPE)
    X_test = sequence.pad_sequences(X_test, maxlen=max_review_length, padding=PADDING_TYPE)

    y = df[category]
    y = y[y != '-']
    from sklearn.preprocessing import LabelEncoder
    le = LabelEncoder()
    y = le.fit_transform(y)

    y_test = df_test[category]
    y_test = y_test[y_test != '-']
    y_test = le.transform(y_test)

    return X, y, X_test, y_test

def get_ote_dataset(tokenizer_path='../we/tokenizer.pkl'):
    def read_data_from_file(path):
        data = {
            'all' : [],
            'sentences' : [],
            'list_of_poss' : [],
            'list_of_is_aspects' : [],
            'list_of_iobs' : [],
            'raw' : []
        }
        with open(path, "r") as f:
            tokens, words, poss, is_aspects, iob_aspects = [], [], [], [], []
            for line in f:
                line = line.rstrip()
                if line:
                    token = tuple(line.split())
                    words.append(token[0])
                    poss.append(token[1])
                    is_aspects.append(token[2])
                    iob_aspects.append(token[6])
                    tokens.append(token)
                else:
                    data['all'].append(tokens)
                    data['sentences'].append(words)
                    data['list_of_poss'].append(poss)
                    data['list_of_is_aspects'].append(is_aspects)
                    data['list_of_iobs'].append(iob_aspects)
                    data['raw'].append(" ".join(words))
                    tokens, words, poss, is_aspects, iob_aspects = [], [], [], [], []
        return data

    train_data = read_data_from_file('data/train_data.txt')
    test_data = read_data_from_file('data/test_data.txt')
                
    df = pd.DataFrame(train_data)
    df_test = pd.DataFrame(test_data)

    """
        Calculate Metrics
    """
    from scipy import stats
    sentence_lengths = []
    for sentence in train_data['sentences']:
        sentence_lengths.append(len(sentence))
    print("max :", np.max(sentence_lengths))
    print("min :", np.min(sentence_lengths))
    print("mean:", np.mean(sentence_lengths))
    print("mode:", stats.mode(sentence_lengths))

    tokenizer = get_tokenizer(tokenizer_path)

    """
        Create X and Y
    """
    # df = df.sample(frac=1, random_state=7)

    X = train_data['raw']
    X_test = test_data['raw']
    X = tokenizer.texts_to_sequences(X)
    X_test = tokenizer.texts_to_sequences(X_test)

    # truncate and pad input sequences
    max_review_length = 81
    X = sequence.pad_sequences(X, maxlen=max_review_length, padding='post', value=-1)
    X_test = sequence.pad_sequences(X_test, maxlen=max_review_length, padding='post', value=-1)

    dum = ['O ASPECT-B ASPECT-I']
    iob_tokenizer = Tokenizer(filters='')
    iob_tokenizer.fit_on_texts(dum)

    from keras.utils import to_categorical
    y_raw = [" ".join(x) for x in df['list_of_iobs']]
    y_raw = iob_tokenizer.texts_to_sequences(y_raw)
    y = sequence.pad_sequences(y_raw, maxlen=max_review_length, padding='post', value=1.)

    y_test_raw = [" ".join(x) for x in df_test['list_of_iobs']]
    y_test_raw = iob_tokenizer.texts_to_sequences(y_test_raw)
    y_test = sequence.pad_sequences(y_test_raw, maxlen=max_review_length, padding='post', value=1.)

    y = to_categorical(y)
    y_test = to_categorical(y_test)

    y = y[:,:,1:]
    y_test = y_test[:,:,1:]

    return X, y, X_test, y_test

def get_sample_weight(X_train, y_train):
    import math

    labels_dict = {}
    for sents in y_train:
        for sent in sents:
            for word in sent:
                for i, value in enumerate(sent):
                    if value == 1:
                        labels_dict[i] = labels_dict.get(i,0) + 1

    def create_class_weight(labels_dict,mu=0.15, **kwargs):
        total = np.sum(list(labels_dict.values()))
        keys = labels_dict.keys()
        class_weight = dict()
        threshold = kwargs.get('threshold', 1.)
        scale = kwargs.get('scale', 1.)
        
        """ OLD """
        # for key in keys:
        #     score = (total-float(labels_dict[key]))/total * scale
        #     class_weight[key] = score if score > threshold else threshold

        # return class_weight

        """ NEW """
        for key in keys:
            score = math.log(mu*total/float(labels_dict[key]))
            class_weight[key] = score if score > 1.0 else 1.0

        return class_weight

    max_review_length = 81

    class_weight = create_class_weight(labels_dict, 2.5, threshold=0.1, scale=5)
    sample_weight = np.zeros((len(y_train), max_review_length))

    for i, samples in enumerate(sample_weight):
        for j, _ in enumerate(samples):
            if X_train[i][j] == -1: #if is padding
                sample_weight[i][j] = 0
            else:
                for k, value in enumerate(y_train[i][j]):
                    if value == 1.:
                        sample_weight[i][j] = class_weight[k]
                        break
    return sample_weight

def get_sentence_end_index(X):
    end = []
    for j, datum in enumerate(X):
        end.append(datum.shape[0])
        for i, token in enumerate(datum):
            if token == -1:
                end[j] = i
                break
    return np.array(end)