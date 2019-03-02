import utils
import nltk
import pickle
import numpy as np
import math
from pathlib import Path

class NaiveBayes():
    def __init__(self, train_filename, pickle_word_prob='', pickle_class_prob=''):
        self.file_name = train_filename
        self.word_prob = {}
        self.class_prob = [0, 0, 0, 0, 0]
        self.pickle_class_prob = pickle_class_prob
        self.pickle_word_prob = pickle_word_prob

    def create_word_count(self):
        review_iterator = utils.json_reader(self.file_name)
        for review in review_iterator:
            text = review['text']
            text = text.split(' ')
            print(text)
            rating = int(review['stars'])
            self.class_prob[rating-1] += 1
            for word in text:
                if word not in self.word_prob:
                    self.word_prob[word] = [0, 0, 0, 0, 0]
                self.word_prob[word][rating-1] += 1

    def calc_word_prob(self):
        for word in self.word_prob.keys():
            for class_idx in range(5):
                self.word_prob[word][class_idx] = math.log(self.word_prob[word][class_idx] / self.class_prob[class_idx])
        class_sum = sum(self.class_prob)
        for class_idx in range(5):
            self.class_prob[class_idx] = math.log(self.class_prob[class_idx]/class_sum)

    def fit(self):
        word_prob = Path(self.pickle_word_prob)
        class_prob = Path(self.pickle_class_prob)
        if word_prob.is_file():
            self.word_prob = pickle.load(open(word_prob, 'rb'))
            self.class_prob = pickle.load(open(class_prob, 'rb'))
        else:
            self.create_word_count()
            self.calc_word_prob()

    def model(self, processed_text):
        num_classes = len(self.class_prob)
        prob_class = np.zeros(num_classes)
        for rating in range(0, num_classes):
            for word in processed_text:
                prob_class[rating] += self.word_prob[word][rating]
            prob_class[rating] += self.class_prob[rating]
        return prob_class

    def predict(self, test_filename):
        review_iterator = utils.json_reader(test_filename)
        iter = 0
        for review in review_iterator:
            text = review['text']
            rating = review['stars']
            print(rating, self.model(text))
            iter += 1
            if iter == 10:
                break

def main(train_filename, test_filname):
    word_prob = Path('pickle_word_prob')
    count_class = Path('pickle_count_class')

    naive_bayes = NaiveBayes(train_filename, word_prob, count_class)
    naive_bayes.fit()

    print(naive_bayes.word_prob)

if __name__ == '__main__':
    train_filename = '../temp/A2/data/train.json'
    test_filename = '../temp/A2/data/test.json'
    main(train_filename, test_filename)