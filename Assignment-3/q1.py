import sys
import pandas
import utils


def part_a(
    train_filename,
    test_filename,
    val_filename
):
    df = pandas.read_csv(train_filename)
    df = df.iloc[1:, :].astype(float)
    cols_to_binarize = ['X1', 'X5', 'X12', 'X13', 'X14', 'X15', 'X16', 'X17'
                        , 'X18', 'X19', 'X20', 'X21', 'X22', 'X23']
    
    df = utils.binarize_median(df, cols_to_binarize)
    print(df.columns)
    X_train = df.drop('Y', axis=1)
    Y_train = df['Y']

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
