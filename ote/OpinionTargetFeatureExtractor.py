import sys
try:
    from constants import Const
    sys.path.insert(0, Const.ROOT)
except:
    sys.path.insert(0, '..')
    from constants import Const

import dill
import re
import string
from sklearn.base import BaseEstimator, TransformerMixin
from keras.models import load_model
from keras.preprocessing import sequence
from keras import backend as K
from RnnOpinionTargetExtractor import RNNOpinionTargetExtractor
import utils
import numpy as np

def get_shape(word):
    word_shape = 'other'
    if re.match('[0-9]+(\.[0-9]*)?|[0-9]*\.[0-9]+$', word):
        word_shape = 'number'
    elif re.match('\W+$', word):
        word_shape = 'punct'
    elif re.match('[A-Z][a-z]+$', word):
        word_shape = 'capitalized'
    elif re.match('[A-Z]+$', word):
        word_shape = 'uppercase'
    elif re.match('[a-z]+$', word):
        word_shape = 'lowercase'
    elif re.match('[A-Za-z]+$', word):
        word_shape = 'mixedcase'
    elif re.match('__.+__$', word):
        word_shape = 'wildcard'
 
    return word_shape

def ner_features(tokens, i, history, proba, clusters, included_features = ['rnn_proba', 'word', 'pos', 'cluster'], included_words=[-2,-1,0,1,2]):
    """
    `tokens`  = a POS-tagged sentence [(w1, t1), ...]
    `i`   = the index of the token we want to extract features for
    `history` = the previous predicted IOB tags
    """
    # Pad the sequence with placeholders
    tokens = [('__START2__', '__START2__'), ('__START1__', '__START1__')] + list(tokens) + [('__END1__', '__END1__'), ('__END2__', '__END2__')]
    history = ['__START2__', '__START1__'] + list(history)

    # shift the i with 2, to accommodate the padding
    i += 2
    features = {}
    proba = proba.tolist()
    proba = [[0, 0, 0], [0, 0, 0]] + proba + [[0, 0, 0], [0, 0, 0]]
    clusters = clusters.tolist()
    clusters = [-2, -2] + clusters + [-2, -2]

    # Start extracting
    word, pos = tokens[i]
    pword, ppos = tokens[i - 1]
    ppword, pppos = tokens[i - 2]
    nword, npos = tokens[i + 1]
    nnword, nnpos = tokens[i + 2]

    cproba = proba[i]
    pproba = proba[i - 1]
    ppproba = proba[i - 2]
    nproba = proba[i + 1]
    nnproba = proba[i + 2]

    shape = get_shape(word)
    pshape = get_shape(pword)
    ppshape = get_shape(ppword)
    nshape = get_shape(nword)
    nnshape = get_shape(nnword)

    list_features = {
        'rnn_proba': [
            [
                ('-2proba-O', ppproba[0]),
                ('-2proba-B', ppproba[1]),
                ('-2proba-I', ppproba[2]),
            ],
            
            [
                ('-1proba-O', pproba[0]),
                ('-1proba-B', pproba[1]),
                ('-1proba-I', pproba[2]),
            ],

            [
                ('proba-O', cproba[0]),
                ('proba-B', cproba[1]),
                ('proba-I', cproba[2])
            ],

            [
                ('+1proba-O', nproba[0]),
                ('+1proba-B', nproba[1]),
                ('+1proba-I', nproba[2]),
            ],

            [
                ('+2proba-O', nnproba[0]),
                ('+2proba-B', nnproba[1]),
                ('+2proba-I', nnproba[2]),
            ],
        ],

        'cluster': [
            [('-2cluster', clusters[i-2])],
            [('-1cluster', clusters[i-1])],
            [('cluster', clusters[i])],
            [('+1cluster', clusters[i+1])],
            [('+2cluster', clusters[i+2])],

            # ('cluster_bigram[-2,-1]', " ".join([str(c) for c in [clusters[i-2], clusters[i-1]]])),
            # ('cluster_bigram[-1,0]', " ".join([str(c) for c in [clusters[i-1], clusters[i]]])),
            # ('cluster_bigram[0,+1]', " ".join([str(c) for c in [clusters[i], clusters[i+1]]])),
            # ('cluster_bigram[+1,+2]', " ".join([str(c) for c in [clusters[i+1], clusters[i+2]]])),
            
            # ('cluster_trigram[-2,-1,0]', " ".join([str(c) for c in [clusters[i-2], clusters[i-1], clusters[i]]])),
            # ('cluster_trigram[-1,0,1]', " ".join([str(c) for c in [clusters[i-1], clusters[i], clusters[i+1]]])),
            # ('cluster_trigram[0,+1,+1]', " ".join([str(c) for c in [clusters[i], clusters[i+1], clusters[i+2]]])),
        ],
        
        'word': [
            [
                ('-2word', ppword),
                ('-2word.lower()', ppword.lower()),
                ('-2word.isupper()', ppword.isupper()),
                ('-2word.istitle()', ppword.istitle()),
                ('-2word.isdigit()', ppword.isdigit()),
            ],
            [
                ('-1word', pword),
                ('-1word.lower()', pword.lower()),
                ('-1word.isupper()', pword.isupper()),
                ('-1word.istitle()', pword.istitle()),
                ('-1word.isdigit()', pword.isdigit()),
            ],
            [
                ('word', word),
                ('word.lower()', word.lower()),
                ('word.isupper()', word.isupper()),
                ('word.istitle()', word.istitle()),
                ('word.isdigit()', word.isdigit()),
            ],

            [
                ('+1word', nword),
                ('+1word.lower()', nword.lower()),
                ('+1word.isupper()', nword.isupper()),
                ('+1word.istitle()', nword.istitle()),
                ('+1word.isdigit()', nword.isdigit()),
            ],

            [
                ('+2word', nnword),
                ('+2word.lower()', nnword.lower()),
                ('+2word.isupper()', nnword.isupper()),
                ('+2word.istitle()', nnword.istitle()),
                ('+2word.isdigit()', nnword.isdigit()),
            ],

            # ('word_bigram[-2,-1]', " ".join([ppword, pword])),
            # ('word_bigram[-1,0]', " ".join([pword, word])),
            # ('word_bigram[0,+1]', " ".join([word, nword])),
            # ('word_bigram[+1,+2]', " ".join([nword, nnword])),
            
            # ('word_trigram[-2,-1,0]', " ".join([ppword, pword, word])),
            # ('word_trigram[-1,0,1]', " ".join([pword, word, nword])),
            # ('word_trigram[0,+1,+1]', " ".join([word, nword, nnword])),

            
        ],

        'shape': [
            [('-2shape', ppshape)],
            [('-1shape', pshape)],
            [('shape', shape)],
            [('+1shape', nshape)],
            [('+2shape', nnshape)],

            # ('shape_bigram[-2,-1]', " ".join([ppshape, pshape])),
            # ('shape_bigram[-1,0]', " ".join([pshape, shape])),
            # ('shape_bigram[0,+1]', " ".join([shape, nshape])),
            # ('shape_bigram[+1,+2]', " ".join([nshape, nnshape])),
            
            # ('shape_trigram[-2,-1,0]', " ".join([ppshape, pshape, shape])),
            # ('shape_trigram[-1,0,1]', " ".join([pshape, shape, nshape])),
            # ('shape_trigram[0,+1,+1]', " ".join([shape, nshape, nnshape])),
        ],

        'pos': [
            [('-2pos', pppos)],
            [('-1pos', ppos)],
            [('pos', pos)],
            [('+1pos', npos)],
            [('+2pos', nnpos)],

            # ('pos_bigram[-2,-1]', " ".join([pppos, ppos])),
            # ('pos_bigram[-1,0]', " ".join([ppos, pos])),
            # ('pos_bigram[0,+1]', " ".join([pos, npos])),
            # ('pos_bigram[+1,+2]', " ".join([npos, nnpos])),
            
            # ('pos_trigram[-2,-1,0]', " ".join([pppos, ppos, pos])),
            # ('pos_trigram[-1,0,1]', " ".join([ppos, pos, npos])),
            # ('pos_trigram[0,+1,+1]', " ".join([pos, npos, nnpos])),
        ],
    }
        
    for feature in included_features:
        for imin2 in included_words:
            i = imin2 + 2
            for feature_name, feature_value in list_features[feature][i]:
                features['{}'.format(feature_name)] = feature_value

    return features

def sent2features(iob_tags, iob_tagged_sentence, proba, clusters, feature_detector, included_features = ['rnn_proba', 'word', 'pos', 'cluster'], included_words=[-2,-1,0,1,2]):
    X_sent = []
    for index in range(len(iob_tagged_sentence)):
        X_sent.append(feature_detector(iob_tagged_sentence, index, proba=proba, history=iob_tags[:index], clusters=clusters, included_features=included_features, included_words=included_words))

    return X_sent

def extract_features(pos_tagged_sentences, iob_tags, feature_detector=ner_features, included_features = ['rnn_proba', 'word', 'pos', 'cluster'], included_words=[-2,-1,0,1,2]):
    """
    Transform a list of tagged sentences into a scikit-learn compatible POS dataset
    :param parsed_sentences:
    :param feature_detector:
    :return:
    """
    tokenizer = utils.get_tokenizer()
    sentences = []
    for pos_tagged_sentence in pos_tagged_sentences:
        sentence, pos = zip(*pos_tagged_sentence)
        sentences.append(sentence)
    sentences = [" ".join(words) for words in sentences]
    
    # GET RNN PROBA
    X_rnn = tokenizer.texts_to_sequences(sentences)
    X_rnn = sequence.pad_sequences(X_rnn, maxlen=81, padding='post', value=-1)

    tags = [
        'ADJ', 'ADP', 'ADV', 'AUX', 'CONJ', 'DET', 'INTJ', 'NOUN', 'NUM', 
        'PART', 'PRON', 'PROPN', 'PUNCT', 'SCONJ', 'SYM', 'VERB', 'X'
    ]
    from keras.preprocessing.text import Tokenizer
    from keras.utils import to_categorical
    from polyglot.text import Text
    pos_tokenizer = Tokenizer()
    pos_tokenizer.fit_on_texts(tags)

    def read_pos_from_sentences(sentences):
        pos = []
        for sent in sentences:
            plg = Text(sent)
            plg.language = 'id'
            _, plg  = zip(*plg.pos_tags)
            pos.append(" ".join(list(plg)))
        pos = pos_tokenizer.texts_to_sequences(pos)
        return pos
    
    pos_rnn = read_pos_from_sentences(sentences)
    pos_rnn = sequence.pad_sequences(pos_rnn, maxlen=81, padding='post', value=-1)
    pos_rnn = to_categorical(pos_rnn)

    # GET CLUSTERS
    list_of_clusters = None
    with open(Const.CLUSTER_ROOT + 'cluster_list_1000.pkl', 'rb') as fi:
        list_of_clusters = dill.load(fi)
    from we.cluster.KMeans import transform
    clusters = transform(X_rnn, list_of_clusters)

    X = []
    K.clear_session()
    ote = RNNOpinionTargetExtractor()
    ote.load_best_model()
    proba = ote.predict([X_rnn, pos_rnn], batch_size = 1)

    for i in range(len(pos_tagged_sentences)):
        X_sent = sent2features(iob_tags[i], pos_tagged_sentences[i], proba[i], clusters[i], feature_detector, included_features=included_features, included_words=included_words)
        X.append(X_sent)

    return X