import pickle
import math
import random
import utils
import utilities
from pathlib import Path
from utilities import getLemmatizedDocument


class NaiveBayes():
    def __init__(
        self,
        train_filename,
        option,
        c=1,
        n_grams=1,
        pickle_word_prob='',
        pickle_class_count='',
        pickle_prior=''
    ):
        self.file_name = train_filename
        self.word_prob = {}
        self.class_word_count = [0, 0, 0, 0, 0]
        self.prior = [0, 0, 0, 0, 0]
        self.process_option = option
        self.pickle_class_count = pickle_class_count
        self.pickle_word_prob = pickle_word_prob
        self.pickle_prior = pickle_prior
        self.c = c
        self.n_grams = n_grams

    def process_text(
        self,
        text,
    ):
        split_text = []
        if self.process_option == 0:
            split_text = text.split(' ')
        elif self.process_option == 1:
            split_text = utils.getStemmedDocuments(text)
        elif self.process_option == 2:
            split_text = utilities.getLemmatizedDocument(text)
        else:
            print('Invalid Option, returning same output as default option')
            split_text = text.split(' ')
        return utilities.find_ngrams(split_text, self.n_grams)

    def create_word_count(
        self
    ):
        review_iterator = utils.json_reader(self.file_name)
        for review in review_iterator:
            text = review['text']
            processed_text = self.process_text(text)
            rating = int(review['stars'])
            self.prior[rating-1] += 1
            for word in processed_text:
                if word not in self.word_prob:
                    self.word_prob[word] = [0, 0, 0, 0, 0]
                self.word_prob[word][rating-1] += 1
                self.class_word_count[rating-1] += 1

    def calc_word_prob(
        self
    ):
        num_unique_words = len(self.word_prob)
        for word in self.word_prob.keys():
            for class_idx in range(5):
                self.word_prob[word][class_idx] = math.log(
                        (self.word_prob[word][class_idx] + self.c) /
                        (self.class_word_count[class_idx] +
                            self.c * num_unique_words))

        num_reviews = sum(self.prior)
        print('Counts', self.prior)
        for class_idx in range(5):
            self.prior[class_idx] = math.log(self.prior[class_idx]/num_reviews)

    def fit(
        self
    ):
        word_prob = Path(self.pickle_word_prob)
        class_word_count = Path(self.pickle_class_count)
        prior = Path(self.pickle_prior)
        if (word_prob.is_file() and class_word_count.is_file() and
                prior.is_file()):
            self.word_prob = pickle.load(open(word_prob, 'rb'))
            self.class_word_count = pickle.load(open(class_word_count, 'rb'))
            self.prior = pickle.load(open(prior, 'rb'))
            print('Loaded learnt model')
        else:
            self.create_word_count()
            self.calc_word_prob()
            pickle.dump(self.word_prob, open(self.pickle_word_prob, 'wb'))
            pickle.dump(self.class_word_count,
                        open(self.pickle_class_count, 'wb'))
            pickle.dump(self.prior, open(self.pickle_prior, 'wb'))

    def model(
        self,
        processed_text
    ):
        num_classes = len(self.class_word_count)
        num_unique_words = len(self.word_prob)
        prob_class = [0 for _ in range(num_classes)]
        prob_default = [math.log(self.c) -
                        math.log(self.class_word_count[class_idx] +
                        self.c * num_unique_words) 
                        for class_idx in range(num_classes)]
        for rating in range(num_classes):
            prob_class[rating] += self.prior[rating]
            for word in processed_text:
                if word not in self.word_prob:
                    self.word_prob[word] = prob_default
                prob_class[rating] += self.word_prob[word][rating]

        return prob_class.index(max(prob_class)) + 1

    def predict(
        self,
        test_filename
    ):
        review_iterator = utils.json_reader(test_filename)
        y_pred = []
        y = []
        for review in review_iterator:
            text = review['text']
            rating = review['stars']
            processed_text = self.process_text(text)
            pred = self.model(processed_text)
            y_pred.append(pred)
            y.append(int(rating))

        return y, y_pred

    def predict_random(
        self,
        test_filename
    ):
        review_iterator = utils.json_reader(test_filename)
        y_pred = []
        y = []
        for review in review_iterator:
            rating = review['stars']
            pred = random.randint(1, 5)
            y_pred.append(pred)
            y.append(int(rating))

        return y, y_pred

    def predict_majority(
        self,
        test_filename
    ):
        review_iterator = utils.json_reader(test_filename)
        y_pred = []
        y = []
        majority_class = self.prior.index(max(self.prior)) + 1
        for review in review_iterator:
            rating = review['stars']
            pred = majority_class
            y_pred.append(pred)
            y.append(int(rating))

        return y, y_pred
