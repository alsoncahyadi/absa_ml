import sys

sys.path.insert(0, '..')

import dill
from sklearn.base import BaseEstimator, TransformerMixin
from keras.models import load_model

def ner_features(tokens, index, history, included_words = [-2, -1, 0, 1, 2]):
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

    for included_word in included_words:
        word, pos = tokens[index + included_word]
        features['{}:word'.format(included_word)] = word
        features['{}:pos'.format(included_word)] = pos
        features['{}:pos[:2]'.format(included_word)] = pos[:2]
        features['{}:word.lower()'.format(included_word)] = str(word.lower())
        # features['{}:word.isupper()'.format(included_word)] = word.isupper()
        # features['{}:word.istitle()'.format(included_word)] = word.istitle()
        # features['{}:word.isdigit()'.format(included_word)] = word.isdigit()
        if included_word < 0: # previous iob
            features['{}:iob'.format(included_word)] = history[included_word]

    return features

def sent2features(iob_tags, iob_tagged_sentence, feature_detector):
    X_sent = []
    for index in range(len(iob_tagged_sentence)):
        X_sent.append(feature_detector(iob_tagged_sentence, index, history=iob_tags[:index]))

    return X_sent

def extract_features(pos_tagged_sentences, iob_tags, feature_detector=ner_features):
    """
    Transform a list of tagged sentences into a scikit-learn compatible POS dataset
    :param parsed_sentences:
    :param feature_detector:
    :return:
    """
    X = []
    for i in range(len(pos_tagged_sentences)):
        X_sent = sent2features(iob_tags[i], pos_tagged_sentences[i], feature_detector)
        X.append(X_sent)

    return X