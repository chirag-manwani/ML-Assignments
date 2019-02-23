import sys
import nltk
import pickle
sys.path.append('../temp/A2/ass2_data/')
import utils

def create_count_dict(train_filename):
    review_iterator = utils.json_reader(train_filename)
    count_dict = {}
    for review in review_iterator:
        text = review['text']
        rating = int(review['stars'])
        processed_text = utils.getStemmedDocuments(text)
        for word in processed_text:
            if word not in count_dict:
                count_dict[word] = [0, 0, 0, 0, 0]
            count_dict[word][rating] += 1
    return count_dict

def main(train_filename, test_filname):
    count_dict = create_count_dict(train_filename)
    pickle.dump(count_dict, open('count_dict.p', 'wb'))

if __name__ == '__main__':
    train_filename = '../temp/A2/data/train.json'
    test_filename = '../temp/A2/data/test.json'
    main(train_filename, test_filename)


