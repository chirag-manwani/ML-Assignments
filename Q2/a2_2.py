import sys
import pandas
import pickle
import numpy as np
from pathlib import Path
from svm import svm_problem, svm_parameter
from svmutil import svm_train, svm_predict
from itertools import combinations
from utilities import accuracy_score
from SVM import SVM


def get_data(
    data,
    digit_1,
    digit_2
):
    positive_class = data.loc[data.iloc[:, -1] == digit_1]
    negative_class = data.loc[data.iloc[:, -1] == digit_2]

    data = positive_class.append(negative_class)

    X = data.iloc[:, :-1]
    Y = data.iloc[:, -1]

    Y = Y.replace(digit_2, -1)
    Y = Y.replace(digit_1, 1)

    return X.values / 255.0, Y.values.astype(float)


def part_1a(
    train_filename,
    test_filename,
    digit
):
    data = pandas.read_csv(train_filename, header=None)
    X_train, Y_train = get_data(data, digit, (digit+1) % 10)

    svm = SVM(c=1, threshold=1e-4)
    svm.fit(X_train, Y_train)

    data = pandas.read_csv(test_filename, header=None)
    X_test, Y_test = get_data(data, digit, (digit+1) % 10)

    Y_pred = svm.predict(X_test)
    print('Accuracy-', accuracy_score(Y_test, Y_pred))


def part_1b(
    train_filename,
    test_filenames,
    digit
):
    data = pandas.read_csv(train_filename, header=None)
    X_train, Y_train = get_data(data, digit, (digit+1) % 10)

    svm = SVM(c=1, threshold=1e-7, kernel='gaussian')
    svm.fit(X_train, Y_train)

    data = pandas.read_csv(test_filename, header=None)
    X_test, Y_test = get_data(data, digit, (digit+1) % 10)

    Y_pred = svm.predict(X_test)
    print('Accuracy-', accuracy_score(Y_test, Y_pred))


def part_1c(
    train_filename,
    test_filename,
    digit,
    kernel='linear'
):
    data = pandas.read_csv(train_filename, header=None)
    X_train, Y_train = get_data(data, digit, (digit+1) % 10)

    param_str = '-s 0 -t 0 -c 1'
    if kernel == 'gaussian':
        param_str = '-s 0 -t 2 -c 1 -g 0.05'
    svm_prb = svm_problem(Y_train, X_train)
    params = svm_parameter(param_str)
    svm_model = svm_train(svm_prb, params)

    data = pandas.read_csv(test_filename, header=None)
    X_test, Y_test = get_data(data, digit, (digit+1) % 10)

    labels, acc, vals = svm_predict(Y_test, X_test, svm_model)
    print('Accuracy-', acc)


def part_2a(
    train_filename,
    test_filename
):
    num_classes = 10
    svm_model = {}

    classifier_idx = list(combinations(range(num_classes), 2))

    svm_models_file = Path('pickle_files/svm_model')

    if svm_models_file.is_file():
        svm_model = pickle.load(open(svm_models_file, 'rb'))
    else:
        for digits in classifier_idx:
            data = pandas.read_csv(train_filename, header=None)
            X_train, Y_train = get_data(data, digits[0], digits[1])

            svm = SVM(c=1, threshold=1e-4, kernel='gaussian')
            svm.fit(X_train, Y_train)
            svm_model[digits] = svm
    print('MultiClass Classfier Model Learnt')

    data = pandas.read_csv(test_filename, header=None)
    X_test = data.iloc[:, :-1].values / 255
    Y_test = data.iloc[:, -1].values

    votes = np.zeros((X_test.shape[0], num_classes))
    dist = np.zeros((X_test.shape[0], num_classes))
    for digits in classifier_idx:
        svm = svm_model[digits]
        Y_pred, Y_dist = svm.predict(X_test)
        row_idx_1 = np.where(Y_pred == 1)[0]
        row_idx_2 = np.where(Y_pred == -1)[0]
        votes[row_idx_1, digits[0]] += 1
        votes[row_idx_2, digits[1]] += 1
        dist[row_idx_1, digits[0]] += np.abs(Y_dist[row_idx_1])
        dist[row_idx_2, digits[1]] += np.abs(Y_dist[row_idx_2])

    print()
    # Considering only votes
    Y_pred = np.argmax(votes, axis=1)
    print('Accuracy with votes- ', accuracy_score(Y_test, Y_pred))

    # Considering only dist
    Y_pred = np.argmax(dist, axis=1)
    print('Accuracy with dist- ', accuracy_score(Y_test, Y_pred))

    # Considering both

    Y_pred = []
    for votes_t, dist_t in zip(votes, dist):
        max_votes = np.max(votes_t)
        final_idx = np.where(votes_t == max_votes)[0]
        if final_idx.shape[0] != 1:
            dist_max = np.max(dist_t[final_idx])
            final_idx = np.where(dist_t == dist_max)[0]
        Y_pred.append(final_idx[0])
    print('Accuracy with both- ', accuracy_score(Y_test, Y_pred))


def part_2b(
    train_filename,
    test_filename
):
    print('a')


def part_2c(

):
    print('part_2c')


def part_2d(

):
    print('part_2d')


def main(
    train_filename,
    test_filename,
    q_num,
    part,
    digit
):
    if q_num == 0:
        if part == 'a':
            part_1a(train_filename, test_filename, digit)
        elif part == 'b':
            part_1b(train_filename, test_filename, digit)
        elif part == 'c':
            part_1c(train_filename, test_filename, digit, kernel='gaussian')
        else:
            print('Invalid Part number. a-c valid')
    elif q_num == 1:
        if part == 'a':
            part_2a(train_filename, test_filename)
        elif part == 'b':
            part_2b(train_filename, test_filename)
        elif part == 'c':
            part_2c(train_filename, test_filename)
        elif part == 'd':
            part_2d(train_filename, test_filename)
        else:
            print('Invalid Part number. a-d valid')
    else:
        print('Invalid Question number. 0 or 1 valid')
        exit()


if __name__ == '__main__':
    args = sys.argv
    if len(args) != 5:
        print('Incorrect number of arguments. 4 expected.')
        exit()

    train_filename = args[1]
    test_filename = args[2]
    q_num = int(args[3])
    part = args[4]
    digit = 0
    main(train_filename, test_filename, q_num, part, digit)
