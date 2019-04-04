import numpy as np
import math


def binarize_median(
    df,
    columns
):
    for col in columns:
        median = df[col].median()
        df[col] = (df[col] >= median).astype(int)
    return df


def entropy(
    labels
):
    n_labels = len(labels)

    if n_labels <= 1:
        return 0

    _, counts = np.unique(labels, return_counts=True)
    label_prob = counts / n_labels
    n_classes = np.count_nonzero(label_prob)

    if n_classes <= 1:
        return 0

    entropy = 0
    for prob in label_prob:
        entropy -= prob * math.log(prob, 2)

    return entropy


def one_hot_encoder(
    X,
    cols_to_encode,
    col_values
):
    cols_to_leave = list(set(range(X.shape[1])) - set(cols_to_encode))
    X_processed = X[:, cols_to_leave]
    num_rows = X.shape[0]
    for col, values in zip(cols_to_encode, col_values):
        val_to_int = {val: idx for idx, val in enumerate(values)}
        X_new_cols = np.zeros((num_rows, len(values)))
        for row in range(num_rows):
            value = X[row, col]
            X_new_cols[row, val_to_int[value]] = 1
        X_processed = np.hstack((X_processed, X_new_cols))
    return X_processed


def sig(x):
    return 1-(1/(1+math.exp(x))) if x < 0 else 1/(1+math.exp(-x))


def sigmoid(z):
    vec_sig = np.vectorize(sig)
    return vec_sig(z)


def relu(z):
    z[z < 0] = 0
    return z
