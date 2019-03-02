import utils
import nltk
import pickle
import numpy as np
from pathlib import Path

def create_count_dict(train_filename):
    review_iterator = utils.json_reader(train_filename)
    count_dict = {}
    count_class = np.ones(5)
    for review in review_iterator:
        text = review['text']
        rating = int(review['stars'])
        count_class[rating-1] += 1
        processed_text = utils.getStemmedDocuments(text)
        for word in processed_text:
            if word not in count_dict:
                count_dict[word] = np.ones(5)
            count_dict[word][rating-1] += 1
    return count_dict, count_class

def calc_word_log_prob(count_dict, count_class):
    for word in count_dict.keys():
        count_dict[word] = np.log(count_dict[word] / count_class)
    count_class /= np.sum(count_class)
    count_class = np.log(count_class)
    return count_dict, count_class

def predict_class(processed_text, theta_j_y, theta_y):
    num_classes = theta_y.shape[0]
    prob_class = np.zeros(num_classes)
    for rating in range(0, num_classes):
        for word in processed_text:
            prob_class[rating] += theta_j_y[word][rating]
        prob_class[rating] += theta_y[rating]
    return prob_class

def predict(test_filename, theta_j_y, theta_y):
    review_iterator = utils.json_reader(test_filename)
    iter = 0
    for review in review_iterator:
        text = review['text']
        rating = review['stars']
        processed_text = utils.getStemmedDocuments(text)
        print(rating, predict_class(processed_text, theta_j_y, theta_y))
        iter += 1
        if iter == 10:
            break

def main(train_filename, test_filname):
    count_dict = Path('pickle_count_dict')
    count_class = Path('pickle_count_class')

    if count_dict.is_file():
        count_dict = pickle.load(open('pickle_count_dict', 'rb'))
        count_class = pickle.load(open('pickle_count_class', 'rb'))
    else:
        count_dict, count_class = create_count_dict(train_filename)
        pickle.dump(count_dict, open('pickle_count_dict', 'wb'))
        pickle.dump(count_class, open('pickle_count_class', 'wb'))
    
    theta_j_y = Path('pickle_theta_j')
    theta_y = Path('pickle_theta_y')
    if theta_j_y.is_file():
        theta_j_y = pickle.load(open('pickle_theta_j', 'rb'))
        theta_y = pickle.load(open('pickle_theta_y', 'rb'))
    else:
        theta_j_y, theta_y = calc_word_log_prob(count_dict, count_class)
        pickle.dump(theta_j_y, open('pickle_theta_j', 'wb'))
        pickle.dump(theta_y, open('pickle_theta_y', 'wb'))
    predict(train_filename, theta_j_y, theta_y)


if __name__ == '__main__':
    train_filename = '../temp/A2/data/train.json'
    test_filename = '../temp/A2/data/test.json'
    main(train_filename, test_filename)