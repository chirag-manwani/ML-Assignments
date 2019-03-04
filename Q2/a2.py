import utils
import nltk
import pickle
import numpy as np
import math
import random
from pathlib import Path
from sklearn.metrics import confusion_matrix

def accuracy_score(y, y_pred):
        num_test = len(y)
        correct = 0.
        for idx in range(num_test):
            if y_pred[idx] == y[idx]:
                correct += 1
        return correct/num_test

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
        
    def process_text(self, text, option=0):
        if option == 0:
            return text.split(' ')
        elif option == 1:
            return utils.getStemmedDocuments(text)
        else:
            print('Invalid Option, returning same output as default option')
            return text.split(' ')

    def create_word_count(self):
        review_iterator = utils.json_reader(self.file_name)
        for review in review_iterator:
            text = review['text']
            processed_text = self.process_text(text, self.process_option)
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
        print('Counts', self.prior)
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
        prob_default = [-math.log(num_classes) for _ in self.class_word_count]
        for rating in range(num_classes):
            prob_class[rating] += self.prior[rating]
            for word in processed_text:
                if word not in self.word_prob:
                    self.word_prob[word] = prob_default
                prob_class[rating] += self.word_prob[word][rating]

        return prob_class.index(max(prob_class)) + 1

    def predict(self, test_filename):
        review_iterator = utils.json_reader(test_filename)
        y_pred = []
        y = []
        for review in review_iterator:
            text = review['text']
            rating = review['stars']
            processed_text = self.process_text(text, self.process_option)
            pred = self.model(processed_text)
            y_pred.append(pred)
            y.append(int(rating))

        return y, y_pred

    def predict_random(self, test_filename):
        review_iterator = utils.json_reader(test_filename)
        y_pred = []
        y = []
        for review in review_iterator:
            rating = review['stars']
            pred = random.randint(1, 5)
            y_pred.append(pred)
            y.append(int(rating))

        return y, y_pred

    def predict_majority(self, test_filename):
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

def main(train_filename, test_filname):
    word_prob = 'pickle_word_prob_p'
    class_word_count = 'pickle_class_word_count_p'
    prior = 'pickle_prior_p'

    naive_bayes = NaiveBayes(train_filename, 1, word_prob, class_word_count, prior)
    naive_bayes.fit()

    # y, y_pred = naive_bayes.predict(train_filename)
    # print(accuracy_score(y, y_pred))

    # cf_mat = confusion_matrix(y, y_pred)
    # print(cf_mat)

    y, y_pred = naive_bayes.predict(test_filename)
    print(accuracy_score(y, y_pred))
    
    cf_mat = confusion_matrix(y, y_pred)
    print(cf_mat)

    y, y_pred = naive_bayes.predict_random(test_filename)
    print(accuracy_score(y, y_pred))

    cf_mat = confusion_matrix(y, y_pred)
    print(cf_mat)

    y, y_pred = naive_bayes.predict_majority(test_filename)
    print(accuracy_score(y, y_pred))

    cf_mat = confusion_matrix(y, y_pred)
    print(cf_mat)

if __name__ == '__main__':
    train_filename = '../temp/A2/data/train.json'
    test_filename = '../temp/A2/data/test.json'
    main(train_filename, test_filename)