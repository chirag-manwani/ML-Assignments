import utils
import nltk
import pickle
import math
import random
import sys
from pathlib import Path
from sklearn.metrics import confusion_matrix
from naive_bayes import NaiveBayes


def part_a(train_filename, test_filename):
    word_prob = 'pickle_files/pickle_word_prob'
    class_word_count = 'pickle_files/pickle_class_word_count'
    prior = 'pickle_files/pickle_prior'

    naive_bayes = NaiveBayes(
                        train_filename,
                        0,
                        word_prob,
                        class_word_count,
                        prior
                    )
    naive_bayes.fit()

    y, y_pred = naive_bayes.predict(train_filename)
    print('Accuracy over Training set- ', accuracy_score(y, y_pred))

    y, y_pred = naive_bayes.predict(test_filename)
    print('Accuracy over Testing set- ', accuracy_score(y, y_pred))


def main(train_filename, test_filname, part):
    word_prob = 'pickle_word_prob_p'
    class_word_count = 'pickle_class_word_count_p'
    prior = 'pickle_prior_p'

    naive_bayes = NaiveBayes(train_filename, 1, word_prob, class_word_count,
                             prior)
    naive_bayes.fit()
    print('Training complete')
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
    args = sys.argv
    if len(args) != 4:
        print('Incorrect number of arguments. 3 expected.')
        exit()

    train_filename = args[1]
    test_filename = args[2]
    part = args[3]
    main(train_filename, test_filename, part)