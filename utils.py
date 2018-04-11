from keras.preprocessing import sequence
import pandas as pd
import numpy as np
import dill


def get_tokenizer():
    """
        Load Tokenizer
    """
    # Make Tokenizer (load or from dataset)
    from keras.preprocessing.text import Tokenizer
    tokenizer = Tokenizer()

    with open('../we/tokenizer.pkl', 'rb') as fi:
        tokenizer = dill.load(fi)
    return tokenizer

def get_ce_dataset():
    tokenizer = get_tokenizer()
    
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

def get_spc_dataset(category):
    tokenizer = get_tokenizer()

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