import sys
import pandas
import utils
import numpy as np
from NN import NN


def part_a(
    train_filename,
    test_filename
):
    data = pandas.read_csv(train_filename, header=None)
    X_train = data.drop(10, axis=1).values
    Y_train = data[10].values

    cols_to_encode = range(10)
    col_values = [[1, 2, 3, 4], [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13],
                  [1, 2, 3, 4], [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13],
                  [1, 2, 3, 4], [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13],
                  [1, 2, 3, 4], [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13],
                  [1, 2, 3, 4], [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13]]
    X_train = utils.one_hot_encoder(X_train, cols_to_encode, col_values)
    Y_train = utils.one_hot_encoder(np.reshape(Y_train, (Y_train.shape[0], 1)),
                                    [0], [range(10)])
    return X_train, Y_train


def part_b(
    train_filename,
    test_filename
):
    X_train, Y_train = part_a(train_filename, test_filename)
    nn = NN(85, 10, [100, 50])
    nn.fit(X_train, Y_train)


def part_c(
    train_filename,
    test_filename
):
    print('part-c')


def part_d(
    train_filename,
    test_filename
):
    print('part-d')


def main(
    train_filename,
    test_filname,
    part
):
    if part == 'a':
        part_a(train_filename, test_filename)
    elif part == 'b':
        part_b(train_filename, test_filename)
    elif part == 'c':
        part_c(train_filename, test_filename)
    elif part == 'd':
        part_d(train_filename, test_filename)
    # elif part == 'e':
    #     part_e(train_filename, test_filname)
    # elif part == 'f':
    #     part_e(train_filename, test_filname)
    # elif part == 'g':
    #     part_g(train_filename, test_filname)
    else:
        print('Invalid question part. a-f Valid')
        exit()


if __name__ == '__main__':
    args = sys.argv
    if len(args) != 4:
        print('Incorrect number of arguments. 3 expected.')
        exit()
    train_filename = args[1]
    test_filename = args[2]
    part = args[3]
    main(train_filename, test_filename, part)
