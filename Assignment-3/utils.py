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
