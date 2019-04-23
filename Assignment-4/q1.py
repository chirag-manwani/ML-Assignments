import sys
import numpy as np
import utils
import pickle
from svm import svm_problem, svm_parameter
from svmutil import svm_train, svm_predict, svm_save_model
from sklearn.metrics import f1_score


def main(
    args
):
    # train_path = args[0]
    # pca_path = args[1]
    train_path = args[0]
    kernel = args[1]

    # utils.pca_fit(train_path, pca_path)
    # print('PCA Fit Complete')
    
    # utils.create_episode_data(train_path, pca_path)
    # print('Episode Data Transformed')
    
    # utils.create_training_data(train_path)
    # print('Episode Training Data created')
    
    X = pickle.load(open(train_path, 'rb'))
    np.random.shuffle(X)

    X_train = X[:10000, :250]
    Y_train = X[:10000, 250]

    X_test = X[100000:150000, :250]
    Y_test = X[100000:150000, 250]

    param_str = '-s 0 -t 0 -c 1'
    if kernel == 'gaussian':
        param_str = '-s 0 -t 2 -c 1 -g 0.05'
    
    svm_prb = svm_problem(Y_train, X_train)
    params = svm_parameter(param_str)
    print('Training')
    svm_model = svm_train(svm_prb, params)
    print('Trained')

    Y_pred, acc, _ = svm_predict(Y_test, X_test, svm_model)
    
    print(f1_score(Y_test, Y_pred, pos_label=1))

if __name__ == "__main__":
    args = sys.argv
    main(args[1:])

# '/media/cmkmanwani/hdd/chirag/test'