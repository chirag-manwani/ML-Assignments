import sys
import pandas
from cvxopt import matrix


def read_features(filename):
    data = pandas.read_csv(filename, header=None)
    X = matrix(data.iloc[:, :-1].values)
    Y = matrix(data.iloc[:, -1].values)
    return X, Y


def part_1a(
    train_filename,
    test_filename
):
    X, Y = read_features(train_filename)
    print('part_1a')


def part_1b(

):
    print('part_1b')


def part_1c(

):
    print('part_1c')


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
):
    if q_num == 0:
        if part == 'a':
            part_1a()
        elif part == 'b':
            part_1b()
        elif part == 'c':
            part_1c()
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
    q_num = args[3]
    part = args[4]
    main(train_filename, test_filename, q_num, part)
