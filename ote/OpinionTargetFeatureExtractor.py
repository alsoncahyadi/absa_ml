import sys

sys.path.insert(0, '..')

import dill
from sklearn.base import BaseEstimator, TransformerMixin
from keras.models import load_model
from keras.preprocessing import sequence
from RnnOpinionTargetExtractor import RNNOpinionTargetExtractor
import utils
import numpy as np

def ner_features(tokens, index, history, proba, included_words = [-2, -1, 0, 1, 2], included_features=[0, 1, 2, 3, 4, 5, 6 ,7, 8, 9, 10]):
    """
    `tokens`  = a POS-tagged sentence [(w1, t1), ...]
    `index`   = the index of the token we want to extract features for
    `history` = the previous predicted IOB tags
    """
    # Pad the sequence with placeholders
    tokens = [('__START2__', '__START2__'), ('__START1__', '__START1__')] + list(tokens) + [('__END1__', '__END1__'), ('__END2__', '__END2__')]
    history = ['__START2__', '__START1__'] + list(history)

    # shift the index with 2, to accommodate the padding
    index += 2
    features = {}
    proba = proba.tolist()
    proba = [[0, 0, 0], [0, 0, 0]] + proba + [[0, 0, 0], [0, 0, 0]]

    for included_word in included_words:
        word, pos = tokens[index + included_word]
        current_proba = [int(new_proba * 1000000) for new_proba in proba[index + included_word]]
        current_proba = proba[index + included_word]
        list_features = [
            ('word', word),
            ('pos', pos),
            ('pos[:2]', pos[:2]),
            ('word.lower()', word.lower()),
            ('word.isupper()', word.isupper()),
            ('word.istitle()', word.istitle()),
            ('word.isdigit()', word.isdigit()),
            ('proba-O', current_proba[0]),
            ('proba-B', current_proba[1]),
            ('proba-I', current_proba[2]),
        ]
        if included_word < 0: # previous iob
            list_features.append(('iob', history[included_word]))
        
        for i in included_features:
            try:
                feature_name = list_features[i][0]
                feature_value = list_features[i][1]
                features['{}:{}'.format(included_word, feature_name)] = feature_value
            except:
                pass

    return features

def sent2features(iob_tags, iob_tagged_sentence, proba, feature_detector, included_words = [-2, -1, 0, 1, 2], included_features=[0, 1, 2, 3, 4]):
    X_sent = []
    for index in range(len(iob_tagged_sentence)):
        X_sent.append(feature_detector(iob_tagged_sentence, index, proba=proba, history=iob_tags[:index], included_features=included_features, included_words=included_words))

    return X_sent

def extract_features(pos_tagged_sentences, iob_tags, feature_detector=ner_features, included_words = [-2, -1, 0, 1, 2], included_features=[0, 1, 2, 3, 4]):
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
    
    X_rnn = tokenizer.texts_to_sequences(sentences)
    X_rnn = sequence.pad_sequences(X_rnn, maxlen=81, padding='post', value=-1)

    X = []
    ote = RNNOpinionTargetExtractor()
    ote.load_best_model()
    proba = ote.predict(X_rnn, batch_size = 1)

    for i in range(len(pos_tagged_sentences)):
        X_sent = sent2features(iob_tags[i], pos_tagged_sentences[i], proba[i], feature_detector, included_words=included_words, included_features=included_features)
        X.append(X_sent)

    return X