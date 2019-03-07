def accuracy_score(y, y_pred):
    num_test = len(y)
    correct = 0.
    for idx in range(num_test):
        if y_pred[idx] == y[idx]:
            correct += 1
    return correct/num_test
