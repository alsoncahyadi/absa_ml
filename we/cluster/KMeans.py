import sys
sys.path.insert(0, '../..')
import dill
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import CountVectorizer
from utils import get_ce_dataset, get_spc_dataset, get_ote_dataset
import numpy as np
import os

OOV_CLUSTER = -1
OOV_TOKEN = 15001

def transform(X, cluster_list):
    new_X = np.zeros(X.shape, dtype=int)
    for i, datum in enumerate(X):
        for j, token in enumerate(datum):
            if token == OOV_TOKEN:
                new_X[i][j] = OOV_CLUSTER
            else:
                new_X[i][j] = cluster_list[token]
    return new_X

if __name__ == "__main__":
    if len(sys.argv) == 1:
        print("argv: 'train' or 'load'")
        exit()
    elif sys.argv[1] == 'train':
        IS_TRAIN = True
    elif sys.argv[1] == 'load':
        IS_TRAIN = False
    else:
        print("argv: 'train' or 'load'")
        exit()

    embedding_matrix = None
    with open('../embedding_matrix.pkl', 'rb') as fi:
        embedding_matrix = dill.load(fi)
    
    print("Embedding Matrix shape:", embedding_matrix.shape)

    name = "1000"

    if IS_TRAIN:
        clus = KMeans(init='k-means++', n_clusters=1000, n_init=10, verbose=1, n_jobs=-1)

        clus.fit(embedding_matrix)

        with open('kmeans_{}.pkl'.format(name), 'wb') as fo:
            dill.dump(clus, fo)
    else:
        clus = None
        with open('kmeans_{}.pkl'.format(name), 'rb') as fi:
            clus = dill.load(fi)

    # cluster_list = clus.predict(embedding_matrix)


    # with open('cluster_list_{}.pkl'.format(name), 'wb') as fo:
    #     dill.dump(cluster_list, fo)

    """
        Save 
            Category Extraction
        Count Vectorizer Vocabulary
    """
    with open('cluster_list_{}.pkl'.format(name), 'rb') as fi:
        cluster_list = dill.load(fi)
    
    os.chdir('../../category_extraction')
    X, _, _, _ = get_ce_dataset()
    X = transform(X, cluster_list)
    bag = CountVectorizer(ngram_range=(1, 2))
    X_text = []
    for datum in X: 
        X_text.append(" ".join([str(token) for token in datum if token != OOV_CLUSTER]))
    bag.fit(X_text)

    print(len(bag.vocabulary_.items()))
    with open('data/count_vectorizer_vocabulary_cluster.pkl', 'wb') as fo:
        dill.dump(bag.vocabulary_, fo)

    """
        Save 
            Opinion Target Extraction
        Count Vectorizer Vocabulary
    """
    os.chdir('../ote')
    X, _, _, _ = get_ote_dataset()
    X = transform(X, cluster_list)
    bag = CountVectorizer(ngram_range=(1, 2))
    X_text = []
    for datum in X: 
        X_text.append(" ".join([str(token) for token in datum if token != OOV_CLUSTER]))
    bag.fit(X_text)
    
    print(len(bag.vocabulary_.items()))
    with open('data/count_vectorizer_vocabulary_cluster.pkl', 'wb') as fo:
        dill.dump(bag.vocabulary_, fo)

    """
        Save 
            Sentiment Polarity Classification
        Count Vectorizer Vocabulary
    """
    os.chdir('../sentiment_polarity')

    categories = ['food', 'service', 'place', 'price']
    for category in categories:
        X, _, _, _ = get_spc_dataset(category)
        X = transform(X, cluster_list)
        bag = CountVectorizer(ngram_range=(1, 2))
        X_text = []
        for datum in X: 
            X_text.append(" ".join([str(token) for token in datum if token != OOV_CLUSTER]))
        bag.fit(X_text)
        
        print(category, len(bag.vocabulary_.items()))
        with open('data/count_vectorizer_vocabulary_cluster_{}.pkl'.format(category), 'wb') as fo:
            dill.dump(bag.vocabulary_, fo)