import utils
import nltk
import pickle
import numpy as np
from pathlib import Path

def create_count_dict(train_filename):
    review_iterator = utils.json_reader(train_filename)
    count_dict = {}
    count_class = [0, 0, 0, 0, 0]
    for review in review_iterator:
        text = review['text']
        rating = int(review['stars'])
        count_class[rating-1] += 1
        processed_text = utils.getStemmedDocuments(text)
        for word in processed_text:
            if word not in count_dict:
                count_dict[word] = [0, 0, 0, 0, 0]
            count_dict[word][rating-1] += 1
    return count_dict, count_class

def calc_word_prob(count_dict, count_class):
    for word in count_dict.keys():
        count_dict[word] = count_dict[word] / count_class
    count_class /= np.sum(count_class)
    return count_dict, count_class

def main(train_filename, test_filname):
    count_dict = Path('pickle_count_dict')
    count_class = Path('pickle_count_class')

    if count_dict.is_file():
        count_dict = pickle.load(open('pickle_count_dict', 'rb'))
        count_class = pickle.load(open('pickle_count_class', 'rb'))
    else:
        count_dict, count_class = create_count_dict(train_filename)
    
    theta_j_y, theta_y = calc_word_prob(count_dict, count_class)
    

if __name__ == '__main__':
    train_filename = '../temp/A2/data/train.json'
    test_filename = '../temp/A2/data/test.json'
    main(train_filename, test_filename)