import sys
import pandas
from svm import svm_problem, svm_parameter
from svmutil import svm_train, svm_predict
from utilities import accuracy_score
from SVM import SVM


def get_data(data, digit):
    positive_label = digit
    negative_label = (digit+1) % 10

    positive_class = data.loc[data.iloc[:, -1] == positive_label]
    negative_class = data.loc[data.iloc[:, -1] == negative_label]

    data = positive_class.append(negative_class)

    X = data.iloc[:, :-1]
    Y = data.iloc[:, -1]

    Y = Y.replace(positive_label, 1)
    Y = Y.replace(negative_label, -1)

    return X.values / 255.0, Y.values.astype(float)


def part_1a(
    train_filename,
    test_filename,
    digit
):
    data = pandas.read_csv(train_filename, header=None)
    X_train, Y_train = get_data(data, digit)

    svm = SVM(c=1, threshold=1e-4)
    svm.fit(X_train, Y_train)

    data = pandas.read_csv(test_filename, header=None)
    X_test, Y_test = get_data(data, digit)

    Y_pred = svm.predict(X_test)
    print('Accuracy-', accuracy_score(Y_test, Y_pred))


def part_1b(
    train_filename,
    test_filenames,
    digit
):
    data = pandas.read_csv(train_filename, header=None)
    X_train, Y_train = get_data(data, digit)

    svm = SVM(c=1, threshold=1e-4, kernel='gaussian')
    svm.fit(X_train, Y_train)

    data = pandas.read_csv(test_filename, header=None)
    X_test, Y_test = get_data(data, digit)

    Y_pred = svm.predict(X_test)
    print('Accuracy-', accuracy_score(Y_test, Y_pred))


def part_1c(
    train_filename,
    test_filename,
    digit,
    kernel='linear'
):
    data = pandas.read_csv(train_filename, header=None)
    X_train, Y_train = get_data(data, digit)

    param_str = '-s 0 -t 2 -c 1'
    if kernel == 'gaussian':
        param_str += ' -g 0.05'
    svm_prb = svm_problem(Y_train, X_train)
    params = svm_parameter(param_str)
    svm_model = svm_train(svm_prb, params)

    data = pandas.read_csv(test_filename, header=None)
    X_test, Y_test = get_data(data, digit)

    labels, acc, vals = svm_predict(Y_test, X_test, svm_model)
    print('Accuracy-', acc)


def part_2a(

):
    print('part_2a')


def part_2b(

):
    print('part_2b')


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
            part_1c(train_filename, test_filename, digit)
        else:
            print('Invalid Part number. a-c valid')
    elif q_num == 1:
        if part == 'a':
            part_2a()
        elif part == 'b':
            part_2b()
        elif part == 'c':
            part_2c()
        elif part == 'd':
            part_2d()
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
    digit = 7
    main(train_filename, test_filename, q_num, part, digit)
