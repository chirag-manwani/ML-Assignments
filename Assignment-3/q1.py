import sys
import pandas
import utils
import pickle
from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from pathlib import Path

from DTree import DTree


def part_a(
    train_filename,
    test_filename,
    val_filename
):
    df_train = pandas.read_csv(train_filename, skiprows=[1])
    df_train = df_train.iloc[:, 1:]

    df_test = pandas.read_csv(test_filename, skiprows=[1])

    df_val = pandas.read_csv(val_filename, skiprows=[1])

    cont_cols = ['X1', 'X5', 'X12', 'X13', 'X14', 'X15', 'X16', 'X17',
                 'X18', 'X19', 'X20', 'X21', 'X22', 'X23']

    df_train = utils.binarize_median(df_train, cont_cols)
    df_test = utils.binarize_median(df_test, cont_cols)
    df_val = utils.binarize_median(df_val, cont_cols)

    X_train = df_train.drop('Y', axis=1)
    Y_train = df_train['Y']

    X_test = df_test.drop('Y', axis=1)
    Y_test = df_test['Y']

    X_val = df_val.drop('Y', axis=1)
    Y_val = df_val['Y']

    d_tree = None
    pickle_file = Path('pickle_a')
    if pickle_file.is_file():
        d_tree = pickle.load(open(pickle_file, 'rb'))
    else:
        d_tree = DTree([], gain_threshold=1e-3, purity_threshold=0.99)
        d_tree.fit(X_train, Y_train)
        pickle.dump(d_tree, open(pickle_file, 'wb'))

    Y_pred = d_tree.predict(X_train)
    print(accuracy_score(Y_train, Y_pred))

    Y_pred = d_tree.predict(X_test)
    print(accuracy_score(Y_test, Y_pred))

    Y_pred = d_tree.predict(X_val)
    print(accuracy_score(Y_val, Y_pred))


def part_c(
    train_filename,
    test_filename,
    val_filename
):
    df_train = pandas.read_csv(train_filename, skiprows=[1])
    df_train = df_train.iloc[:, 1:]

    df_test = pandas.read_csv(test_filename, skiprows=[1])

    df_val = pandas.read_csv(val_filename, skiprows=[1])

    cont_cols = ['X1', 'X5', 'X12', 'X13', 'X14', 'X15', 'X16', 'X17',
                 'X18', 'X19', 'X20', 'X21', 'X22', 'X23']

    X_train = df_train.drop('Y', axis=1)
    Y_train = df_train['Y']

    X_test = df_test.drop('Y', axis=1)
    Y_test = df_test['Y']

    X_val = df_val.drop('Y', axis=1)
    Y_val = df_val['Y']

    d_tree = None
    pickle_file = Path('pickle_c')
    if pickle_file.is_file():
        d_tree = pickle.load(open(pickle_file, 'rb'))
    else:
        d_tree = DTree(cont_cols, gain_threshold=1e-3, purity_threshold=0.99)
        d_tree.fit(X_train, Y_train)
        pickle.dump(d_tree, open(pickle_file, 'wb'))

    Y_pred = d_tree.predict(X_train)
    print(accuracy_score(Y_train, Y_pred))

    Y_pred = d_tree.predict(X_test)
    print(accuracy_score(Y_test, Y_pred))

    Y_pred = d_tree.predict(X_val)
    print(accuracy_score(Y_val, Y_pred))


def part_d(
    train_filename,
    test_filename,
    val_filename
):
    df_train = pandas.read_csv(train_filename, skiprows=[1])
    df_train = df_train.iloc[:, 1:]

    df_test = pandas.read_csv(test_filename, skiprows=[1])
    df_test = df_test.iloc[:, 1:]

    df_val = pandas.read_csv(val_filename, skiprows=[1])
    df_val = df_val.iloc[:, 1:]

    # cont_cols = ['X1', 'X5', 'X12', 'X13', 'X14', 'X15', 'X16', 'X17',
    #              'X18', 'X19', 'X20', 'X21', 'X22', 'X23']

    # df_train = utils.binarize_median(df_train, cont_cols)
    # df_test = utils.binarize_median(df_test, cont_cols)
    # df_val = utils.binarize_median(df_val, cont_cols)

    X_train = df_train.drop('Y', axis=1).values
    Y_train = df_train['Y'].values

    X_test = df_test.drop('Y', axis=1).values
    Y_test = df_test['Y'].values

    X_val = df_val.drop('Y', axis=1).values
    Y_val = df_val['Y'].values

    d_tree = DecisionTreeClassifier(
        criterion='gini',
        min_samples_leaf=10,
        min_samples_split=5,
        max_depth=None)
    d_tree.fit(X_train, Y_train)

    Y_pred = d_tree.predict(X_train)
    print(accuracy_score(Y_train, Y_pred))

    Y_pred = d_tree.predict(X_test)
    print(accuracy_score(Y_test, Y_pred))

    Y_pred = d_tree.predict(X_val)
    print(accuracy_score(Y_val, Y_pred))


def part_e(
    train_filename,
    test_filename,
    val_filename
):
    df_train = pandas.read_csv(train_filename, skiprows=[1])
    df_train = df_train.iloc[:, 1:]

    df_test = pandas.read_csv(test_filename, skiprows=[1])
    df_test = df_test.iloc[:, 1:]

    df_val = pandas.read_csv(val_filename, skiprows=[1])
    df_val = df_val.iloc[:, 1:]

    X_train = df_train.drop('Y', axis=1).values
    Y_train = df_train['Y'].values

    X_test = df_test.drop('Y', axis=1).values
    Y_test = df_test['Y'].values

    X_val = df_val.drop('Y', axis=1).values
    Y_val = df_val['Y'].values

    cols_to_encode = [1, 2, 3, 5, 6, 7, 8, 9, 10]
    col_values = [[1, 2], [0, 1, 2, 3, 4, 5, 6], [0, 1, 2, 3],
                  [-2, -1, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
                  [-2, -1, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
                  [-2, -1, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
                  [-2, -1, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
                  [-2, -1, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
                  [-2, -1, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9]]

    X_train = utils.one_hot_encoder(X_train, cols_to_encode, col_values)
    X_test = utils.one_hot_encoder(X_test, cols_to_encode, col_values)
    X_val = utils.one_hot_encoder(X_val, cols_to_encode, col_values)

    d_tree = DecisionTreeClassifier(
        criterion='gini',
        min_samples_leaf=10,
        min_samples_split=5,
        max_depth=None)
    d_tree.fit(X_train, Y_train)

    Y_pred = d_tree.predict(X_train)
    print(accuracy_score(Y_train, Y_pred))

    Y_pred = d_tree.predict(X_test)
    print(accuracy_score(Y_test, Y_pred))

    Y_pred = d_tree.predict(X_val)
    print(accuracy_score(Y_val, Y_pred))


def part_f(
    train_filename,
    test_filename,
    val_filename
):

    df_train = pandas.read_csv(train_filename, skiprows=[1])
    df_train = df_train.iloc[:, 1:]

    df_test = pandas.read_csv(test_filename, skiprows=[1])
    df_test = df_test.iloc[:, 1:]

    df_val = pandas.read_csv(val_filename, skiprows=[1])
    df_val = df_val.iloc[:, 1:]

    X_train = df_train.drop('Y', axis=1).values
    Y_train = df_train['Y'].values

    X_test = df_test.drop('Y', axis=1).values
    Y_test = df_test['Y'].values

    X_val = df_val.drop('Y', axis=1).values
    Y_val = df_val['Y'].values

    cols_to_encode = [1, 2, 3, 5, 6, 7, 8, 9, 10]
    col_values = [[1, 2], [0, 1, 2, 3, 4, 5, 6], [0, 1, 2, 3],
                  [-2, -1, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
                  [-2, -1, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
                  [-2, -1, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
                  [-2, -1, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
                  [-2, -1, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
                  [-2, -1, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9]]

    X_train = utils.one_hot_encoder(X_train, cols_to_encode, col_values)
    X_test = utils.one_hot_encoder(X_test, cols_to_encode, col_values)
    X_val = utils.one_hot_encoder(X_val, cols_to_encode, col_values)

    d_tree = RandomForestClassifier(
        max_depth=None,
        n_estimators=23,
        random_state=0,
        max_features=10,
        bootstrap=True)
    d_tree.fit(X_train, Y_train)

    Y_pred = d_tree.predict(X_train)
    print(accuracy_score(Y_train, Y_pred))

    Y_pred = d_tree.predict(X_test)
    print(accuracy_score(Y_test, Y_pred))

    Y_pred = d_tree.predict(X_val)
    print(accuracy_score(Y_val, Y_pred))


def main(
    train_filename,
    test_filname,
    val_filename,
    part
):
    if part == 'a':
        part_a(train_filename, test_filename, val_filename)
    # elif part == 'b':
    #     part_b(train_filename, test_filename)
    elif part == 'c':
        part_c(train_filename, test_filename, val_filename)
    elif part == 'd':
        part_d(train_filename, test_filname, val_filename)
    elif part == 'e':
        part_e(train_filename, test_filname, val_filename)
    elif part == 'f':
        part_f(train_filename, test_filname, val_filename)
    else:
        print('Invalid question part. a-f Valid')
        exit()


if __name__ == '__main__':
    args = sys.argv
    if len(args) != 5:
        print('Incorrect number of arguments. 4 expected.')
        exit()

    train_filename = args[1]
    test_filename = args[2]
    val_filename = args[3]
    part = args[4]
    main(train_filename, test_filename, val_filename, part)
