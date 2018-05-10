import sys

sys.path.insert(0, '..')

import dill
from sklearn.base import BaseEstimator, TransformerMixin
from keras.models import load_model

def ner_features(tokens, index, history, included_words = [-2, -1, 0, 1, 2], included_features=[0, 1, 2, 3, 4, 5, 6 ,7]):
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
        list_features = [
            ('word', word),
            ('pos', pos),
            ('pos[:2]', pos[:2]),
            ('word.lower()', word.lower()),
            ('word.isupper()', word.isupper()),
            ('word.istitle()', word.istitle()),
            ('word.isdigit()', word.isdigit()),
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
            # features['{}:word.isupper()'.format(included_word)] = word.isupper()
            # features['{}:word.istitle()'.format(included_word)] = word.istitle()
            # features['{}:word.isdigit()'.format(included_word)] = word.isdigit()

    return features

def sent2features(iob_tags, iob_tagged_sentence, feature_detector, included_words = [-2, -1, 0, 1, 2], included_features=[0, 1, 2, 3, 4]):
    X_sent = []
    for index in range(len(iob_tagged_sentence)):
        X_sent.append(feature_detector(iob_tagged_sentence, index, history=iob_tags[:index], included_features=included_features, included_words=included_words))

    return X_sent

def extract_features(pos_tagged_sentences, iob_tags, feature_detector=ner_features, included_words = [-2, -1, 0, 1, 2], included_features=[0, 1, 2, 3, 4]):
    """
    Transform a list of tagged sentences into a scikit-learn compatible POS dataset
    :param parsed_sentences:
    :param feature_detector:
    :return:
    """
    X = []
    for i in range(len(pos_tagged_sentences)):
        X_sent = sent2features(iob_tags[i], pos_tagged_sentences[i], feature_detector, included_words=included_words, included_features=included_features)
        X.append(X_sent)

    return X