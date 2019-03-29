import sys
from sklearn.metrics import confusion_matrix
from sklearn.metrics import f1_score
from naive_bayes import NaiveBayes
from utilities import accuracy_score


def part_a(
        train_filename,
        test_filename
):
    word_prob = 'pickle_files/pickle_word_prob'
    class_word_count = 'pickle_files/pickle_class_word_count'
    prior = 'pickle_files/pickle_prior'

    naive_bayes = NaiveBayes(
                        train_filename,
                        option=0,
                        c=1,
                        n_grams=1,
                        pickle_word_prob=word_prob,
                        pickle_class_count=class_word_count,
                        pickle_prior=prior
                    )
    naive_bayes.fit()

    y, y_pred = naive_bayes.predict(train_filename)
    print('Accuracy over Training set- ', accuracy_score(y, y_pred))

    y, y_pred = naive_bayes.predict(test_filename)
    print('Accuracy over Testing set- ', accuracy_score(y, y_pred))

    return y, y_pred


def part_b(train_filename, test_filename):
    word_prob = 'pickle_files/pickle_word_prob'
    class_word_count = 'pickle_files/pickle_class_word_count'
    prior = 'pickle_files/pickle_prior'

    naive_bayes = NaiveBayes(
                        train_filename,
                        option=0,
                        c=1,
                        n_grams=1,
                        pickle_word_prob=word_prob,
                        pickle_class_count=class_word_count,
                        pickle_prior=prior
                    )
    naive_bayes.fit()

    y, y_pred = naive_bayes.predict_random(test_filename)
    print('Random Prediction Accuracy- ', accuracy_score(y, y_pred))

    y, y_pred = naive_bayes.predict_majority(test_filename)
    print('Majority Prediction Accuracy- ', accuracy_score(y, y_pred))


def part_c(train_filename, test_filename):
    word_prob = 'pickle_files/pickle_word_prob'
    class_word_count = 'pickle_files/pickle_class_word_count'
    prior = 'pickle_files/pickle_prior'

    naive_bayes = NaiveBayes(
                        train_filename,
                        option=0,
                        c=1,
                        n_grams=1,
                        pickle_word_prob=word_prob,
                        pickle_class_count=class_word_count,
                        pickle_prior=prior
                    )
    naive_bayes.fit()

    y, y_pred = naive_bayes.predict(test_filename)
    print('Accuracy over Testing set- ', accuracy_score(y, y_pred))

    cf_mat = confusion_matrix(y, y_pred)
    print(cf_mat)


def part_d(train_filename, test_filename):
    word_prob = 'pickle_files/pickle_word_prob_d'
    class_word_count = 'pickle_files/pickle_class_word_count_d'
    prior = 'pickle_files/pickle_prior_d'

    naive_bayes = NaiveBayes(
                        train_filename,
                        option=1,
                        c=1,
                        n_grams=1,
                        pickle_word_prob=word_prob,
                        pickle_class_count=class_word_count,
                        pickle_prior=prior
                    )
    naive_bayes.fit()

    y, y_pred = naive_bayes.predict(train_filename)
    print('Accuracy over Training set- ', accuracy_score(y, y_pred))

    y, y_pred = naive_bayes.predict(test_filename)
    print('Accuracy over Testing set- ', accuracy_score(y, y_pred))


def part_e(train_filename, test_filename):
    word_prob = 'pickle_files/pickle_word_prob_e'
    class_word_count = 'pickle_files/pickle_class_word_count_e'
    prior = 'pickle_files/pickle_prior_e'

    naive_bayes = NaiveBayes(
                        train_filename,
                        option=2,
                        c=1,
                        n_grams=2,
                        pickle_word_prob=word_prob,
                        pickle_class_count=class_word_count,
                        pickle_prior=prior
                    )
    naive_bayes.fit()

    # y, y_pred = naive_bayes.predict(train_filename)
    # print('Accuracy over Training set- ', accuracy_score(y, y_pred))

    y_true, y_pred = naive_bayes.predict(test_filename)
    print('Accuracy over Testing set- ', accuracy_score(y_true, y_pred))

    score = f1_score(y_true, y_pred, average='macro')
    print('F1 Score', score)


def part_g(train_filename, test_filname):
    word_prob = 'pickle_files/pickle_word_prob_g'
    class_word_count = 'pickle_files/pickle_class_word_count_g'
    prior = 'pickle_files/pickle_prior_g'

    naive_bayes = NaiveBayes(
                        train_filename,
                        option=2,
                        c=1,
                        n_grams=2,
                        pickle_word_prob=word_prob,
                        pickle_class_count=class_word_count,
                        pickle_prior=prior
                    )
    naive_bayes.fit()

    # y, y_pred = naive_bayes.predict(train_filename)
    # print('Accuracy over Training set- ', accuracy_score(y, y_pred))

    y, y_pred = naive_bayes.predict(test_filename)
    print('Accuracy over Testing set- ', accuracy_score(y, y_pred))


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
        part_d(train_filename, test_filname)
    elif part == 'e':
        part_e(train_filename, test_filname)
    elif part == 'f':
        part_e(train_filename, test_filname)
    elif part == 'g':
        part_g(train_filename, test_filname)
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
