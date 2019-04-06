import sys
import pandas
import utils
import numpy as np
import matplotlib.pyplot as plt

from NN import NN


def part_a(
    train_filename,
    test_filename,
    one_hot_train_filename,
    one_hot_test_filename
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

    data = np.hstack((X_train, Y_train))
    np.savetxt(one_hot_train_filename, data.astype(int), delimiter=',',
               fmt='%d')

    data = pandas.read_csv(test_filename, header=None)
    X_test = data.drop(10, axis=1).values
    Y_test = data[10].values

    X_test = utils.one_hot_encoder(X_test, cols_to_encode, col_values)
    Y_test = utils.one_hot_encoder(np.reshape(Y_test, (Y_test.shape[0], 1)),
                                   [0], [range(10)])
    data = np.hstack((X_test, Y_test))
    np.savetxt(one_hot_test_filename, data.astype(int), delimiter=',',
               fmt='%d')


def part_b(
    config_file,
    train_filename,
    test_filename
):
    nI, nO, batch_size, arch_list, activation, lr_type \
        = utils.read_config(config_file)
    data = pandas.read_csv(train_filename, header=None)
    X_train = data.iloc[:, :85].values
    Y_train = data.iloc[:, 85:].values
    nn = NN(nI, nO, arch_list, activation_function=activation)
    nn.fit(X_train, Y_train, lr=1, epochs=1000,
           batch_size=batch_size, adaptive=lr_type, prints=True)


def part_c(
    config_file,
    train_filename,
    test_filename,
    layers=1
):
    nI, nO, batch_size, _, activation, lr_type \
        = utils.read_config(config_file)
    hidden_layer_units = [5, 10, 15, 20, 25]

    data = pandas.read_csv(train_filename, header=None)
    X_train = data.iloc[:, :85].values
    Y_train = data.iloc[:, 85:].values

    data = pandas.read_csv(test_filename, header=None)
    X_test = data.iloc[:, :85].values
    Y_test = data.iloc[:, 85:].values

    train_acc = []
    test_acc = []
    for units in hidden_layer_units:
        nn = NN(nI, nO, [units] * layers, activation_function=activation)
        nn.fit(X_train, Y_train, lr=1, epochs=250,
               batch_size=batch_size, adaptive=lr_type, prints=True)

        train_acc.append(nn.error(X_train, Y_train) * 100)
        test_acc.append(nn.error(X_test, Y_test) * 100)
        print('Done', units, train_acc[-1], test_acc[-1])

    print('Train Set Accuracy', train_acc)
    print('Test Set Accuracy', test_acc)

    ax = plt.gca()
    ax.plot(hidden_layer_units, test_acc, label='Test Set Accuracy')
    ax.plot(hidden_layer_units, train_acc, label='Train Set Accuracy')

    ax.set_xlabel('No. of Hidden units')
    ax.set_ylabel('Accuracy in %')

    ax.set_xticks(hidden_layer_units)

    ax.legend()
    plt.show()


def part_d(
    config_file,
    train_filename,
    test_filename
):
    part_c(config_file, train_filename, test_filename, layers=2)


def part_e(
    config_file,
    train_filename,
    test_filename
):
    part_c(config_file, train_filename, test_filename)
    part_c(config_file, train_filename, test_filename, layers=2)


def main(
    args
):
    part = args[-1]
    if part == 'a':
        part_a(args[0], args[1], args[2], args[3])
    elif part == 'b':
        part_b(args[0], args[1], args[2])
    elif part == 'c':
        part_c(args[0], args[1], args[2])
    elif part == 'd':
        part_d(args[0], args[1], args[2])
    elif part == 'e':
        part_d(args[0], args[1], args[2])
    # elif part == 'f':
    #     part_e(train_filename, test_filname)
    else:
        print('Invalid question part. a-f Valid')
        exit()


if __name__ == '__main__':
    main(sys.argv[1:])
