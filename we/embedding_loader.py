"""
    Prepare Text
"""
texts = None
with open('sswe/vocab_full_500.txt', 'r') as fi:
    texts = fi.readlines()
texts = [x.strip() for x in texts]

print(texts[:5])
print("Length:", len(texts))
print("Unique:", len(set(texts)))

from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
import numpy as np
import dill

"""
    Make Tokenizer
"""

num_words = 15000

tokenizer = Tokenizer(num_words=num_words-1, filters='', oov_token='UNK')
tokenizer.fit_on_texts(texts)
sequences = tokenizer.texts_to_sequences(texts)
sequences = tokenizer.texts_to_sequences(['UNK f;alwenf', 'yang beribu'])
print(sequences)
print(texts[30000])

# tokenizer = Tokenizer()
# with open('tokenizer.pkl', 'rb') as fi:
# 	tokenizer = dill.load(fi)

word_index = tokenizer.word_index
print('Found %s unique tokens.' % len(word_index))

with open('tokenizer.pkl', 'wb') as fo:
    dill.dump(tokenizer, fo)

"""
    Load Embedding from txt file
"""

embeddings_index = {}

with open('sswe/vectors_full_500.txt') as fi:
    next(fi)
    for line in fi:
        values = line.split()
        word = values[0]
        coefs = np.asarray(values[1:], dtype='float32')
        embeddings_index[word] = coefs

# with open('embeddings_index.pkl', 'wb') as fo:
# 	dill.dump(embeddings_index, fo)

# with open('embeddings_index.pkl', 'rb') as fi:
# 	embeddings_index = dill.load(fi)

print('Found %s word vectors.' % len(embeddings_index))


"""
    Make own embedding matrix
"""

EMBEDDING_DIM = 500
MAX_SEQUENCE_LENGTH = 150

embedding_matrix = np.zeros((num_words, EMBEDDING_DIM))

# with open('embedding_matrix.pkl', 'rb') as fi:
# 	embedding_matrix = dill.load(fi)

# print(embedding_matrix.shape)
oov=0
for word, i in word_index.items():
    if i >= num_words: break

    embedding_vector = embeddings_index.get(word)
    if embedding_vector is not None:
        embedding_matrix[i] = embedding_vector
    else:
        # words not found in embedding index will be all-zeros.
        oov += 1
        print(word, "is not found!")

print('OOV:', oov)

""" OOV HANDLING """
# OOV = matrix of 0

with open('embedding_matrix.pkl', 'wb') as fo:
    dill.dump(embedding_matrix, fo)
print("LENGTH", len(embedding_matrix))

"""
    Make Embedding layer
"""

from keras.layers.embeddings import Embedding

embedding_layer = Embedding(num_words,
                            EMBEDDING_DIM,
                            weights=[embedding_matrix],
                            input_length=MAX_SEQUENCE_LENGTH,
                            trainable=False)

from keras import Sequential
model = Sequential()
model.add(embedding_layer)

data = [[15000] * 150]
data = np.array(data)
print(data.shape)
