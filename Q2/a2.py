import sys
import nltk
sys.path.append('../temp/A2/ass2_data/')
import utils

def create_dict_count(train_filename):
    review_iterator = utils.json_reader(train_filename)
    dict_count = {}
    for review in review_iterator:
        text = review['text']
        rating = int(review['stars'])
        processed_text = utils.getStemmedDocuments(text)
        for word in processed_text:
    
            break


def main(train_filename, test_filname):
    create_dict_count(train_filename)

if __name__ == '__main__':
    train_filename = '../temp/A2/data/train.json'
    test_filename = 'test.json'
    main(train_filename, test_filename)

