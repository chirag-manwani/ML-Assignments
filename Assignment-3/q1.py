import sys
import pandas
import utils
from sklearn.metrics import accuracy_score
from DTree import DTree


def part_a(
    train_filename,
    test_filename,
    val_filename
):
    df = pandas.read_csv(train_filename, skiprows=[1])
    df = df.iloc[:, 1:]
    cont_cols = ['X1', 'X5', 'X12', 'X13', 'X14', 'X15', 'X16', 'X17',
                 'X18', 'X19', 'X20', 'X21', 'X22', 'X23']

    df = utils.binarize_median(df, cont_cols)
    # print(df)
    X_train = df.drop('Y', axis=1)
    Y_train = df['Y']

    d_tree = DTree([])
    d_tree.fit(X_train, Y_train)
    Y_pred = d_tree.predict(X_train)
    print(accuracy_score(Y_train, Y_pred))


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
    # elif part == 'c':
    #     part_c(train_filename, test_filename)
    # elif part == 'd':
    #     part_d(train_filename, test_filname)
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
    if len(args) != 5:
        print('Incorrect number of arguments. 3 expected.')
        exit()

    train_filename = args[1]
    test_filename = args[2]
    val_filename = args[3]
    part = args[4]
    main(train_filename, test_filename, val_filename, part)
