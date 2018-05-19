from keras.preprocessing import sequence
from keras.preprocessing.text import Tokenizer
import pandas as pd
import numpy as np
import dill
from constants import Const
from polyglot.text import Text

def is_none(x):
    if type(x).__name__ == 'None':
        return True
    return False

def get_tokenizer():
    """
        Load Tokenizer
    """
    # Make Tokenizer (load or from dataset)
    from keras.preprocessing.text import Tokenizer
    tokenizer = Tokenizer()

    with open(Const.TOKENIZER_PATH, 'rb') as fi:
        tokenizer = dill.load(fi)
    return tokenizer

def get_ce_dataset():
    tokenizer = get_tokenizer()
    
    """
        Construct X and y
    """
    df = pd.read_csv(Const.CE_ROOT + "data/train_data.csv", delimiter=";", header=0, encoding = "ISO-8859-1")
    df_test = pd.read_csv(Const.CE_ROOT + "data/test_data.csv", delimiter=";", header=0, encoding = "ISO-8859-1")

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

def get_spc_dataset(category):
    tokenizer = get_tokenizer()

    """
        Construct X and y
    """
    df = pd.read_csv(Const.SPC_ROOT + "data/train_data_3.csv", delimiter=";", header=0, encoding = "ISO-8859-1")
    df_test = pd.read_csv(Const.SPC_ROOT + "data/test_data_3.csv", delimiter=";", header=0, encoding = "ISO-8859-1")

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

def get_ote_dataset():
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

    train_data = read_data_from_file(Const.OTE_ROOT + 'data/train_data_fixed.txt')
    test_data = read_data_from_file(Const.OTE_ROOT + 'data/test_data_fixed.txt')
                
    df = pd.DataFrame(train_data)
    df_test = pd.DataFrame(test_data)

    """
        Calculate Metrics
    """
    # from scipy import stats
    # sentence_lengths = []
    # for sentence in train_data['sentences']:
    #     sentence_lengths.append(len(sentence))
    # print("max :", np.max(sentence_lengths))
    # print("min :", np.min(sentence_lengths))
    # print("mean:", np.mean(sentence_lengths))
    # print("mode:", stats.mode(sentence_lengths))

    tokenizer = get_tokenizer()

    from polyglot.text import Text
    from keras.utils import to_categorical
    tags = [
        'ADJ', 'ADP', 'ADV', 'AUX', 'CONJ', 'DET', 'INTJ', 'NOUN', 'NUM', 
        'PART', 'PRON', 'PROPN', 'PUNCT', 'SCONJ', 'SYM', 'VERB', 'X'
    ]
    pos_tokenizer = Tokenizer()
    pos_tokenizer.fit_on_texts(tags)

    def read_pos_from_raw(data):
        pos = []
        for sent in data['raw']:
            plg = Text(sent)
            plg.language = 'id'
            _, plg  = zip(*plg.pos_tags)
            pos.append(" ".join(list(plg)))
        pos = pos_tokenizer.texts_to_sequences(pos)
        return pos
    
    pos = read_pos_from_raw(train_data)
    pos_test = read_pos_from_raw(test_data)

    """
        Create X and Y
    """
    X = train_data['raw']
    X_test = test_data['raw']
    X = tokenizer.texts_to_sequences(X)
    X_test = tokenizer.texts_to_sequences(X_test)

    # truncate and pad input sequences
    max_review_length = 81
    PADDING = 'post'
    X = sequence.pad_sequences(X, maxlen=max_review_length, padding=PADDING, value=-1)
    X_test = sequence.pad_sequences(X_test, maxlen=max_review_length, padding=PADDING, value=-1)

    dum = ['O ASPECT-B ASPECT-I']
    iob_tokenizer = Tokenizer(filters='')
    iob_tokenizer.fit_on_texts(dum)

    from keras.utils import to_categorical
    y_raw = [" ".join(x) for x in df['list_of_iobs']]
    y_raw = iob_tokenizer.texts_to_sequences(y_raw)
    y = sequence.pad_sequences(y_raw, maxlen=max_review_length, padding=PADDING, value=1.)

    y_test_raw = [" ".join(x) for x in df_test['list_of_iobs']]
    y_test_raw = iob_tokenizer.texts_to_sequences(y_test_raw)
    y_test = sequence.pad_sequences(y_test_raw, maxlen=max_review_length, padding=PADDING, value=1.)

    pos = sequence.pad_sequences(pos, maxlen=max_review_length, padding='post', value=-1)
    pos_test = sequence.pad_sequences(pos_test, maxlen=max_review_length, padding='post', value=-1)

    y = to_categorical(y)
    y_test = to_categorical(y_test)
    pos = to_categorical(pos)
    pos_test = to_categorical(pos_test)

    y = y[:,:,1:]
    y_test = y_test[:,:,1:]

    return X, y, pos, X_test, y_test, pos_test

def filter_sentence(sentence):
    new_sentence = sentence.replace('-', 'DASH')
    # new_sentence = sentence.replace('~', 'WAVE')
    return new_sentence

def get_crf_ote_dataset():
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

    train_data = read_data_from_file(Const.OTE_ROOT + 'data/train_data_fixed.txt')
    test_data = read_data_from_file(Const.OTE_ROOT + 'data/test_data_fixed.txt')

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

    """
        Create X
    """

    X_raw = train_data['raw']
    X_test_raw = test_data['raw']

    X_pos_tagged = []; X_test_pos_tagged = []

    for sentence in X_raw:
        filtered_sentence = filter_sentence(sentence)
        polyglot_text = Text(filtered_sentence)
        polyglot_text.language = 'id'
        tagged_sentence = polyglot_text.pos_tags
        X_pos_tagged.append(tagged_sentence)

    for sentence in X_test_raw:
        filtered_sentence = filter_sentence(sentence)
        polyglot_text = Text(filtered_sentence)
        polyglot_text.language = 'id'
        tagged_sentence = polyglot_text.pos_tags
        X_test_pos_tagged.append(tagged_sentence)

    """
        Create Y
    """
    y = df['list_of_iobs'].as_matrix()
    y_test = df_test['list_of_iobs'].as_matrix()

    return X_pos_tagged, y, X_test_pos_tagged, y_test

def create_class_weight(labels_dict,mu=0.1, threshold=1., **kwargs):
    import math
    total = np.sum(list(labels_dict.values()))
    keys = labels_dict.keys()
    class_weight = dict()
    scale = kwargs.get('scale', 1.)
    
    """ OLD """
    # for key in keys:
    #     score = (total-float(labels_dict[key]))/total * scale
    #     class_weight[key] = score if score > threshold else threshold

    # return class_weight

    """ NEW """
    for key in keys:
        score = math.log(mu*total/float(labels_dict[key]))
        class_weight[key] = score if score > threshold else threshold

    return class_weight

def get_sample_weight(X_train, y_train, threshold=0.1, mu=2.5):
    labels_dict = {}
    for sents in y_train:
        for sent in sents:
            for word in sent:
                for i, value in enumerate(sent):
                    if value == 1:
                        labels_dict[i] = labels_dict.get(i,0) + 1

    max_review_length = 81

    class_weight = create_class_weight(labels_dict, mu=mu, threshold=threshold, scale=5)
    # class_weight = {0: 1.0, 1: 5.5, 2: 6.5}
    sample_weight = np.zeros((len(y_train), max_review_length))

    learn_first_padding = False
    for i, samples in enumerate(sample_weight):
        first_padding = True
        for j, _ in enumerate(samples):
            if X_train[i][j] == -1: #if is padding
                if first_padding and learn_first_padding:
                    sample_weight[i][j] = class_weight[0]
                    first_padding = False
                else:
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

def time_log(func):
    import time, datetime
    start_time_local = time.strftime("%d-%m-%Y %H:%M:%S", time.localtime())
    print()
    print("===============================")
    print("Started on:", start_time_local)
    print("===============================")
    print()
    start_time = time.time()
    func()
    print()
    print("================================")
    print("DONE IN {}".format(datetime.timedelta(seconds=(int(time.time() - start_time)))))
    print("Started  on:", start_time_local)
    print("Finished on:", time.strftime("%d-%m-%Y %H:%M:%S", time.localtime()))
    print("================================")
    print()
    
def save_object(obj, filename):
    with open(filename, 'wb') as output_file:
        dill.dump(obj, output_file)

def load_object(filename):
    with open(filename, 'rb') as input_file:
        return dill.load(input_file)

def get_entities_from_iob_tagged_tokens(iob_tagged_tokens):
    #entity = (word, tag)
    #example: ('Wonder Woman', 'movie_title', token pos)
    current_entity = None
    current_entity_words = None
    current_entity_name = None
    current_entity_pos = None
    all_entities = []
    for i, (word, tag, iob_tag) in enumerate(iob_tagged_tokens):
        iob = iob_tag[-2:]
        entity_name = iob_tag[:-2]
        if current_entity_words:
            if iob == "I-": #concatenate word if in chunk
                current_entity_words += " " + word
            else: #append to list if chunk stops
                current_entity = (current_entity_words, current_entity_name, current_entity_pos)
                all_entities.append(current_entity)
                current_entity = None
                current_entity_words = None
                current_entity_name = None
                current_entity_pos = None
        if iob == "B-": #beginning of chunk
            current_entity_name = entity_name
            current_entity_words = word
            current_entity_pos = i

    #check if the last word is in chunk
    if current_entity_words:
        current_entity = (current_entity_words, current_entity_name, current_entity_pos)
        all_entities.append(current_entity)
    return all_entities

def get_entities_from_iob_tagged_tokens1(iob_tagged_tokens):
    #entity = (word, tag)
    #example: ('Wonder Woman', 'movie_title', token begin, token end + 1)
    current_entity = None
    current_entity_words = None
    current_entity_name = None
    current_entity_pos = None
    all_entities = []
    for i, (word, tag, iob_tag) in enumerate(iob_tagged_tokens):
        iob = iob_tag[-2:]
        entity_name = iob_tag[:-2]
        if current_entity_words:
            if iob == "I-": #concatenate word if in chunk
                current_entity_words += " " + word
            else: #append to list if chunk stops
                current_entity = (current_entity_words, current_entity_name, current_entity_pos, i)
                all_entities.append(current_entity)
                current_entity = None
                current_entity_words = None
                current_entity_name = None
                current_entity_pos = None
        if iob == "B-": #beginning of chunk
            current_entity_name = entity_name
            current_entity_words = word
            current_entity_pos = i

    #check if the last word is in chunk
    if current_entity_words:
        current_entity = (current_entity_words, current_entity_name, current_entity_pos, len(iob_tagged_tokens))
        all_entities.append(current_entity)
    return all_entities
