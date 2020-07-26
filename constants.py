from enum import Enum
import os

class Const():
    # Paths
    ROOT = os.getenv('ABSA_ML_ROOT')
    CE_ROOT = ROOT + 'category_extraction/'
    OTE_ROOT = ROOT + 'ote/'
    SPC_ROOT = ROOT + 'sentiment_polarity/'
    WE_ROOT = ROOT + 'we/'
    CLUSTER_ROOT = ROOT + 'we/cluster/'
    REVIEWS_ROOT = ROOT + 'reviews/'

    TOKENIZER_PATH = ROOT + 'we/tokenizer.pkl'

    # Numbers
    EMBEDDING_VECTOR_LENGTH = 500
    VOCABULARY_SIZE = 15000

    CATEGORIES = ["food", "service", "price", "place"]
    SENTIMENTS = ["negative", "positive"]

    PADDING = 0
