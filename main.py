from constants import Const
from category_extraction.BinCategoryExtractor import BinCategoryExtractor
from ote.CrfOpinionTargetExtractor import CRFOpinionTargetExtractor
from ote.OpinionTargetFeatureExtractor import extract_features
from sentiment_polarity.SentimentPolarityClassifier import CNNSentimentPolarityClassifier
from polyglot.text import Text
from nltk.tokenize import sent_tokenize
from pprint import pprint
from polyglot.text import Text
from devina.tuple_generator import TupleGenerator
from keras import backend as K
import nltk.data
import utils

class Main():
    """
        1. opinion target extraction
        2. split sentence
        3. category extraction
        4. sentiment polarity classification
        5. generate tuple

        data:
        0 ==> preprocessed (tokenized)
        1 ==> aspects extracted
        2 ==> categorys extracted
        3 ==> sentiments extracted
        4 ==> tuples
        5 ==> ratings
    """
    def __init__(self, sentence_tokenizer='normal'):
        """
            Sentence Tokenizers: normal, punkt
        """
        self.sent_tokenize = sent_tokenize
        if sentence_tokenizer == 'normal':
            pass
        elif sentence_tokenizer == 'punkt':
            self.sent_tokenize = nltk.data.load('tokenizers/punkt/english.pickle').tokenize

        self.tokenizer = utils.get_tokenizer()
        self.raw_reviews = utils.get_raw_test_reviews()
        self.data = []
        for _ in range(5):
            self.data.append([])
        
        self.categories = ['food', 'service', 'price', 'place']
        self.conjunctions = ["tetapi sayangnya", "namun", "tetapi", "walaupun", "akan tetapi", "sayangnya",
                             "hanya sayang", "sayang", "meski", "walau", "but"]

    def _get_aspect_label(self, tokens):
        label = ["ASPECT-B"]
        for i in range(1, len(tokens)):
            label.append("ASPECT-I")
        return label

    def _get_aspects_from_tokens(self, tokens, labels):
        result = []
        indices = [i for i, x in enumerate(labels) if x == "ASPECT-B"]
        for i in indices:
            aspect = tokens[i]
            j = i + 1
            while j < len(tokens):
                if labels[j] == "ASPECT-I":
                    aspect += " " + tokens[j]
                    j += 1
                else:
                    break
            result.append(aspect)
        return result

    def split_sentence(self, sentence, label):
        def parts(list_, indices):
            indices = [0] + indices + [len(list_)]
            return [list_[v:indices[k + 1]] for k, v in enumerate(indices[:-1])]

        tokens = sentence.split()
        indices = []

        for conjunction in self.conjunctions:
            if conjunction in sentence:
                indices += [i for i, x in enumerate(tokens) if x == conjunction]
        indices.sort()

        del_indices = []
        for i in range(1, len(indices)):
            if indices[i] - indices[i - 1] == 1:
                del_indices.append(i)

        for i in del_indices:
            indices.pop(i)

        sentence_partitions = parts(tokens, indices)
        label_partitions = parts(label, indices)
        # if len(sentence_partitions) > 1:
        #     print sentence

        for i in range(1, len(label_partitions)):
            if "ASPECT-B" not in label_partitions[i] and "ASPECT-B" in label_partitions[i - 1]:
                aspects = self._get_aspects_from_tokens(sentence_partitions[i - 1], label_partitions[i - 1])
                last_aspect_tokens = aspects[-1].split()

                sentence_partitions[i].pop(0)
                label_partitions[i].pop(0)
                if sentence_partitions[i][0] in self.conjunctions:
                    sentence_partitions[i].pop(0)
                    label_partitions[i].pop(0)

                sentence_partitions[i] = last_aspect_tokens + sentence_partitions[i]
                label_partitions[i] = self._get_aspect_label(last_aspect_tokens) + label_partitions[i]

        sentence_partitions = [x for x in sentence_partitions if x != []]
        label_partitions = [x for x in label_partitions if x != []]
        # if len(sentence_partitions) > 1:
        #     print sentence_partitions
        #     print label_partitions
        #     print "\n"
        return sentence_partitions, label_partitions

    """ ==========================================================================================="""

    def preprocess(self, skip_sentence_tokenize=False):
        sents_tokenized = []
        if skip_sentence_tokenize:
            sents_tokenized = self.raw_reviews
        else:
            for raw_review in self.raw_reviews:
                sents_tokenized += self.sent_tokenize(raw_review)
        
        word_tokenized = []
        for sent_tokenize in sents_tokenized:
            tmp = Text(sent_tokenize)
            tmp.language = 'id'
            word_tokenized.append(" ".join(tmp.words))
        self.data[0] = word_tokenized
    
    def predict_opinion_targets(self):
        K.clear_session()
        crf_ote = CRFOpinionTargetExtractor()
        crf_ote.load_best_model()
        X_pos = utils.prepare_crf_X(self.data[0])
        X = extract_features(X_pos)
        y_pred = crf_ote.predict(X)
        self.data[1] = y_pred
        return y_pred

    def split_sentences(self):
        tokens = []
        labels = []
        for i in range(len(self.data[0])):
            # sentence = " ".join(self.results[0][i])
            sentence = self.data[0][i]
            splitted_tokens, label = self.split_sentence(sentence, self.data[1][i])

            for j in range(len(splitted_tokens)):
                tokens.append(splitted_tokens[j])
                labels.append(label[j])

        self.data[0] = " ".join(tokens)
        self.data[1] = labels

    def predict_categories(self):
        K.clear_session()
        bin_ce = BinCategoryExtractor(included_features=[0])
        bin_ce.load_estimators()
        X = utils.prepare_ce_X(self.data[0], self.tokenizer)
        y_pred = bin_ce.predict(X)
        self.category_target_names = bin_ce.target_names
        self.data[2] = y_pred
        return y_pred

    def predict_sentiment_polarities(self):
        K.clear_session()
        y_preds = []
        for category in self.category_target_names:
            spc = CNNSentimentPolarityClassifier()
            spc.load_best_model(category)
            X = utils.prepare_ce_X(self.data[0], self.tokenizer)
            y_preds.append(spc.predict(X))
        self.data[3] = y_preds
        return y_preds

    def get_tuples(self):
        tuples = {"food": {"positive": [], "negative": []},
                  "price": {"positive": [], "negative": []},
                  "place": {"positive": [], "negative": []},
                  "service": {"positive": [], "negative": []}}
        tuples_unique = {"food": {"positive": [], "negative": []},
                         "price": {"positive": [], "negative": []},
                         "place": {"positive": [], "negative": []},
                         "service": {"positive": [], "negative": []}}
        tuple_generator = TupleGenerator()

        for i in range(len(self.data[0])):
            aspects = self._get_aspects_from_tokens(self.data[0][i].split(), self.data[1][i])
            category_sentiment = {}
            for j in range(len(self.data[2][i])):
                if self.data[2][i][j] == 1:
                    category = self.category_target_names[j]
                    polarity = 'positive' if self.data[3][j][i] == 1. else 'negative'
                    category_sentiment[category] = polarity

            if len(aspects) > 0 and len(category_sentiment) > 0:
                result = tuple_generator.generate_tuples(aspects, category_sentiment)

                for key in result:
                    for sentiment in result[key]:
                        tuples[key][sentiment] += result[key][sentiment]
                        for item in result[key][sentiment]:
                            if item not in tuples_unique[key][sentiment]:
                                tuples_unique[key][sentiment].append(item)
        
        self.data[4] = {
            'tuples': tuples,
            'tuples_unique': tuples_unique
        }
        return tuples, tuples_unique

    def get_ratings(self):
        tuples = self.data[4]['tuples']
        ratings = {"food": [], "price": [], "place": [], "service": []}

        for category in tuples:
            pos = len(tuples[category]["positive"])
            neg = len(tuples[category]["negative"])
            ratings[category].append((pos * 4 / (pos + neg)) + 1)
            ratings[category].append(round((float(pos) * 4 / (pos + neg)) + 1, 2))

        return ratings

def main():
    m = Main(sentence_tokenizer='normal')
    m.preprocess(skip_sentence_tokenize=True)
    m.predict_opinion_targets()
    m.predict_categories()
    m.predict_sentiment_polarities()
    m.get_tuples()
    print(m.get_ratings())

if __name__ == '__main__':
    utils.time_log(main)