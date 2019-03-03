import utils
import nltk
import pickle
import numpy as np
import math
from pathlib import Path

class NaiveBayes():
    def __init__(self, train_filename, option, pickle_word_prob='', pickle_class_prob='', pickle_prior=''):
        self.file_name = train_filename
        self.word_prob = {}
        self.class_word_count = [1, 1, 1, 1, 1]
        self.prior = [1, 1, 1, 1, 1]
        self.process_option = option
        self.pickle_class_prob = pickle_class_prob
        self.pickle_word_prob = pickle_word_prob
        self.pickle_prior = pickle_prior
        
    def process_text(self, text):
        return text.split()

    def create_word_count(self):
        review_iterator = utils.json_reader(self.file_name)
        for review in review_iterator:
            text = review['text']
            processed_text = self.process_text(text)
            rating = int(review['stars'])
            self.prior[rating-1] += 1
            for word in processed_text:
                if word not in self.word_prob:
                    self.word_prob[word] = [1, 1, 1, 1, 1]
                self.word_prob[word][rating-1] += 1
                self.class_word_count[rating-1] += 1

    def calc_word_prob(self):
        for word in self.word_prob.keys():
            for class_idx in range(5):
                self.word_prob[word][class_idx] = math.log(self.word_prob[word][class_idx] / self.class_word_count[class_idx])
        num_reviews = sum(self.prior)
        for class_idx in range(5):
            self.prior[class_idx] = math.log(self.prior[class_idx]/num_reviews)

    def fit(self):
        word_prob = Path(self.pickle_word_prob)
        class_word_count = Path(self.pickle_class_prob)
        prior = Path(self.pickle_prior)
        if word_prob.is_file() and class_word_count.is_file() and prior.is_file():
            self.word_prob = pickle.load(open(word_prob, 'rb'))
            self.class_word_count = pickle.load(open(class_word_count, 'rb'))
            self.prior = pickle.load(open(prior, 'rb'))
            print('Loaded learnt model')
        else:
            self.create_word_count()
            self.calc_word_prob()
            pickle.dump(self.word_prob, open(self.pickle_word_prob, 'wb'))
            pickle.dump(self.class_word_count, open(self.pickle_class_prob, 'wb'))
            pickle.dump(self.prior, open(self.pickle_prior, 'wb'))

    def model(self, processed_text):
        num_classes = len(self.class_word_count)
        prob_class = [0 for _ in range(num_classes)]
        prob_default = [-math.log(i) for i in self.class_word_count]
        for rating in range(num_classes):
            for word in processed_text:
                prob_class[rating] += self.prior[rating]
                if word not in self.word_prob:
                    self.word_prob[word] = prob_default
                prob_class[rating] += self.word_prob[word][rating]
        return prob_class.index(max(prob_class))

    def predict(self, test_filename):
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

        num_test = len(y)
        correct = 0.
        for idx in range(num_test):
            if y_pred[idx] == y[idx]:
                correct += 1
        accuracy = correct/num_test
        return y_pred, accuracy

def main(train_filename, test_filname):
    word_prob = 'pickle_word_prob'
    class_word_count = 'pickle_class_word_count'
    prior = 'pickle_prior'
    option = 0

    naive_bayes = NaiveBayes(train_filename, option, word_prob, class_word_count, prior)
    naive_bayes.fit()
    y_pred, accuracy = naive_bayes.predict(test_filename)
    print(accuracy)

if __name__ == '__main__':
    train_filename = '../temp/A2/data/train.json'
    test_filename = '../temp/A2/data/test.json'
    main(train_filename, test_filename)