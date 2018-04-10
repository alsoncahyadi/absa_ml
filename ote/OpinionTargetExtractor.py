import sys
sys.path.insert(0, '..')

from MyClassifier import MyClassifier
from sklearn.base import BaseEstimator, ClassifierMixin

from keras.models import Sequential, Input, Model, load_model
from keras.layers.convolutional import Conv1D
from keras.layers import Dense, LSTM, Dropout, Lambda, Bidirectional, TimeDistributed, RepeatVector, RNN
from keras.layers.pooling import AveragePooling1D, MaxPooling1D, GlobalMaxPooling1D
from keras.layers.embeddings import Embedding
from keras.preprocessing import sequence, text
from keras import regularizers, optimizers
from keras.callbacks import ModelCheckpoint

import dill
import numpy as np

import matplotlib.pyplot as plt
import itertools
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split

class RNNOpinionTargetExtractor (MyClassifier):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.WEIGHTS_PATH = 'model/brnn/weights/BRNN.hdf5'
        self.MODEL_PATH = 'model/brnn/BRNN.model'
        self.WE_PATH = '../we/embedding_matrix.pkl'
       
        self.layer_embedding = self._load_embedding(self.WE_PATH, trainable=True, vocabulary_size=15000, embedding_vector_length=500)
        self.rnn_model = None
        for key, value in kwargs.items():
            setattr(self, key, value)
    
    def fit(self, X, y, **kwargs):
        self.rnn_model = self._create_model()
        self.rnn_model.save(self.MODEL_PATH)
        self.rnn_model.summary()
        mode = kwargs.get('mode', 'train_validate_split')
        if mode == "train_validate_split":
            self.rnn_model.fit(
                X, y,
                **kwargs
            )
            self.rnn_model.load_weights(self.WEIGHTS_PATH)
    
    def predict(self, X, **kwargs):
        y_pred = self.rnn_model.predict(X)
        return y_pred
    
    def score(self, X, y, **kwargs):
        from keras.models import load_model
        rnn_model = self._create_model(n_batch=1)
        print(rnn_model.summary())
        rnn_model.load_weights('model/brnn/weights/blstm_weights.hdf5')
        scores = rnn_model.evaluate(X, y, verbose=0)
        print("Test Set Accuracy: %.2f%%" % (scores[1]*100))

        def max_index(cat):
            i_max = -1
            val_max = -1
            for i, y in enumerate(cat):
                if val_max < y:
                    i_max = i
                    val_max = y
            return i_max

        def get_decreased_dimension(y):
            tmp = []
            for y_sents in y:
                for y_tokens in y_sents:
                    tmp.append(y_tokens)
            tmp = np.array(tmp)
            return tmp

        y_pred_raw = rnn_model.predict(X)
        y_pred = []
        for y_pred_raw_sents in y_pred_raw:
            y_pred_sents = []
            for y_pred_raw_tokens in y_pred_raw_sents:
                max_i = max_index(y_pred_raw_tokens)
                y = [0.] * 3 #number of classes to be predicted
                y[max_i] = 1.
                y_pred_sents.append(y)
            y_pred.append(y_pred_sents)
        y_pred = np.array(y_pred)
        # y_pred = np.argmax(get_decreased_dimension(y_pred_raw), axis=1)
        # y_test = np.argmax(get_decreased_dimension(y_test), axis=1)
        # print(y_pred)

        y_pred = get_decreased_dimension(y_pred)
        y_test = get_decreased_dimension(y)
        print(y_test.shape, y_pred.shape)

        from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score
        AVERAGE = None
        print("F1-Score  : {}".format(f1_score(y_test, y_pred, average=AVERAGE)))
        print("Precision : {}".format(precision_score(y_test, y_pred, average=AVERAGE)))
        print("Recall    : {}".format(recall_score(y_test, y_pred, average=AVERAGE)))

        f1_score_macro = f1_score(y_test, y_pred, average='macro')
        print("F1-Score-Macro:", f1_score_macro)

        is_show_confusion_matrix = kwargs.get('show_confusion_matrix', False)
        if is_show_confusion_matrix:
            self.plot_all_confusion_matrix(y, y_pred)
        
        return f1_score_macro

    def _fit_train_validate_split(self, X, y):
        pass

    def _create_model(
        self,
        recurrent_dropout = 0.5,
        dropout_rate = 0.6,
        dense_activation = 'relu',
        dense_l2_regularizer = 0.01,
        activation = 'sigmoid',
        optimizer = "Nadam",
        loss_function = 'binary_crossentropy',

        **kwargs
    ):
    
        MAX_SEQUENCE_LENGTH = kwargs.get("max_sequence_length")
        # Define Architecture
        layer_input = Input(shape=(MAX_SEQUENCE_LENGTH,))
        layer_embedding = self.layer_embedding(layer_input)
        layer_blstm = Bidirectional(LSTM(256, return_sequences=True, recurrent_dropout=recurrent_dropout, stateful=False))(layer_embedding)
        layer_dropout_1 = TimeDistributed(Dropout(0.5, seed=7))(layer_blstm)
        layer_dense_1 = TimeDistributed(Dense(256, activation=dense_activation, kernel_regularizer=regularizers.l2(dense_l2_regularizer)))(layer_dropout_1)
        layer_softmax = TimeDistributed(Dense(3, activation=activation))(layer_dense_1)
        rnn_model = Model(inputs=layer_input, outputs=layer_softmax)

        # Create Optimizer
        optimizer = optimizers.SGD(lr=0.05, momentum=0.9, decay=0.0, nesterov=True)
        rnn_model.compile(loss=loss_function, optimizer=optimizer, metrics=['accuracy'],
                        sample_weight_mode="temporal")
        return rnn_model
        
        # layer_input = Input(shape=(max_review_length,))
        # layer_embedding = kwargs.get('layer_embedding', Embedding(n_words, embedding_vector_length))(layer_input)
        # layer_lstm = LSTM(256, recurrent_dropout=0.2)(layer_embedding)
        # layer_repeat = RepeatVector(max_review_length)(layer_lstm)
        # layer_blstm = Bidirectional(LSTM(256, return_sequences=True, recurrent_dropout=0.0))(layer_repeat)
        # layer_dropout_1 = TimeDistributed(Dropout(0.6, seed=7))(layer_blstm)
        # layer_dense_1 = TimeDistributed(Dense(256, activation='relu', kernel_regularizer=regularizers.l2(0.01)))(layer_dropout_1)
        # layer_softmax = TimeDistributed(Dense(3, activation='softmax'))(layer_dense_1)
        # rnn_model = Model(inputs=layer_input, outputs=layer_softmax)
    
    def _get_features(self, x):
        return x

    def load_weights(self, path):
        self._create_model()
        self.rnn_model.load_weights(path)
    
    def _plot_confusion_matrix(self,
                          cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
        """
        This function prints and plots the confusion matrix.
        Normalization can be applied by setting `normalize=True`.
        """
        
        if normalize:
            cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

        plt.imshow(cm, interpolation='nearest', cmap=cmap)
        plt.title(title)
        plt.colorbar()
        tick_marks = np.arange(len(classes))
        plt.xticks(tick_marks, classes, rotation=45)
        plt.yticks(tick_marks, classes)

        fmt = '.2f' if normalize else 'd'
        thresh = cm.max() / 2.
        for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
            plt.text(j, i, format(cm[i, j], fmt),
                    horizontalalignment="center",
                    color="white" if cm[i, j] > thresh else "black")

        plt.tight_layout()
        plt.ylabel('True label')
        plt.xlabel('Predicted label')
    
    def plot_all_confusion_matrix(self, y_test, y_pred):
        plt.figure()
        self._plot_confusion_matrix(confusion_matrix(y_test.iloc[:,0], y_pred[:,0]), classes=['0', '1'])
        plt.figure()
        self._plot_confusion_matrix(confusion_matrix(y_test.iloc[:,1], y_pred[:,1]), classes=['0', '1'])
        plt.figure()
        self._plot_confusion_matrix(confusion_matrix(y_test.iloc[:,2], y_pred[:,2]), classes=['0', '1'])
        plt.figure()
        self._plot_confusion_matrix(confusion_matrix(y_test.iloc[:,3], y_pred[:,3]), classes=['0', '1'])
        plt.show()


class CategoryFeatureExtractor (BaseEstimator):
    def __init__(self):
        pass

    def fit(self, x, y=None):
        raise NotImplementedError
    
    def transform(self):
        raise NotImplementedError

def main():
    import pandas as pd
    import numpy as np

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
    maximum_sentence_length = 0
    sentence_lengths = []
    for sentence in train_data['sentences']:
        sentence_lengths.append(len(sentence))
    print("max :", np.max(sentence_lengths))
    print("min :", np.min(sentence_lengths))
    print("mean:", np.mean(sentence_lengths))
    print("mode:", stats.mode(sentence_lengths))

    """
        Make Tokenizer
    """
    # Make Tokenizer (load or from dataset)
    from keras.preprocessing.text import Tokenizer
    tokenizer = Tokenizer()
    with open('../we/tokenizer.pkl', 'rb') as fi:
        tokenizer= dill.load(fi)
    # Print tokenizer detail
    len(tokenizer.word_counts)

    """
        Create X and Y
    """
    df = df.sample(frac=1, random_state=7)

    train_validation_split_at = 800

    X = train_data['raw']
    X_test = test_data['raw']
    X = tokenizer.texts_to_sequences(X)
    X_test = tokenizer.texts_to_sequences(X_test)

    # truncate and pad input sequences
    max_review_length = 81
    X = sequence.pad_sequences(X, maxlen=max_review_length, padding='post', value=-1)
    X_test = sequence.pad_sequences(X_test, maxlen=max_review_length, padding='post', value=-1)

    X_train = X[:800]
    X_validate = X[800:]

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

    y_train = y[:train_validation_split_at]
    y_validate = y[train_validation_split_at:]
    # y_train = np.delete(y_train, np.s_[:,:,3], 1)

    print(y_train.shape)

    print(np.isnan(X).any())
    print(np.isnan(y).any())

    """
        Calculate Sample Weight
    """
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

        for key in keys:
            score = math.log(mu*total/float(labels_dict[key]))
            class_weight[key] = score if score > 1.0 else 1.0

        return class_weight

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
    
    """
        Make and fit the model
    """
    np.random.seed(7)

    checkpointer = ModelCheckpoint(filepath='model/cnn/weights/CNN.hdf5', verbose=1, save_best_only=True)
    ote = RNNOpinionTargetExtractor()

    n_epoch = 20
    # for i in range(n_epoch):
    #     print('Epoch #', i)
    #     model.fit(x=X_train, y=y_train, epochs=1, batch_size=32,
    #           validation_data=(X_validate, y_validate), callbacks=[checkpointer]
    #           ,sample_weight=sample_weight)
    #     model.reset_states()

    ote.fit(X_train, y_train, epochs=n_epoch, batch_size=32,
        validation_data=(X_validate, y_validate), callbacks=[checkpointer]
        ,sample_weight=sample_weight)
    ote.score(X_test, y_test, show_confusion_matrix=True)
    
if __name__ == "__main__":
    main()