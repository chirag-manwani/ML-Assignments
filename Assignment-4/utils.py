import numpy as np
import os
import pickle
import pandas
import keras.backend as K
from PIL import Image
from sklearn.decomposition import PCA
from pathlib import Path
from itertools import combinations


input_shape = (210, 160, 5)
# Preprocessing data utilities


def process_image(
    img_path
):
    # left_crop = 6
    # right_crop = -6
    # top_crop = 40
    # bottom_crop = -40

    img = np.array(Image.open(img_path).convert('L'))
    # img = img[top_crop:bottom_crop, left_crop:right_crop]

    return img.flatten()


def pca_fit(
    train_path,
    pca_path
):
    pca = Path(pca_path)
    if pca.is_file():
        return

    img_list = []
    for root, _, img_files in os.walk(train_path):
        for img_file in sorted(img_files):
            if img_file.endswith('png'):
                img_path = os.path.join(root, img_file)
                flat_image = process_image(img_path)
                img_list.append(flat_image)
    X = np.array(img_list, dtype='float32')
    # scaler = StandardScaler()
    # X = scaler.fit_transform(X)
    X = X - X.mean(axis=0)
    pca = PCA(n_components=50)
    pca = pca.fit(X)
    pickle.dump(pca, open(pca_path, 'wb'))


def create_episode_data(
    train_path,
    pca_path
):
    pca = pickle.load(open(pca_path, 'rb'))
    for root, _, img_files in os.walk(train_path):
        img_list = []
        X_file = Path(os.path.join(root, 'X'))
        if X_file.is_file():
            continue
        for img_file in sorted(img_files):
            if img_file.endswith('png'):
                img_path = os.path.join(root, img_file)
                flat_image = process_image(img_path)
                img_list.append(flat_image)
        X = np.array(img_list)
        X = pca.transform(X)
        root = Path(root)
        pickle.dump(X, open(root.joinpath(X_file), 'wb'))


def get_combinations(
    rew_idx,
    X,
    rewards
):
    X_list = X[rew_idx-6:rew_idx]
    comb_list = list(combinations(X_list, 4))
    for i in range(len(comb_list)):
        comb_list[i] = list(comb_list[i])
        comb_list[i].append(X[rew_idx])
        comb_list[i].append(np.array(rewards[rew_idx]))

    rows = []
    for comb in comb_list:
        rows.append(np.concatenate(comb, axis=0))

    return rows


def create_training_data(
    train_path
):
    X_train = []
    for root, _, _ in os.walk(train_path):
        rewards = Path(os.path.join(root, 'rew.csv'))
        if not rewards.is_file():
            continue
        rewards = pandas.read_csv(rewards, header=None).values
        X = pickle.load(open(os.path.join(root, 'X'), 'rb'))
        one_count = 0
        for rew_idx in range(6, X.shape[0]-1):
            if rewards[rew_idx] == 1:
                X_train += get_combinations(rew_idx, X, rewards)
                one_count += 1
            elif one_count > 0:
                X_train += get_combinations(rew_idx, X, rewards)
                one_count -= 1
    X_train = np.array(X_train)
    root = Path(root)
    pickle.dump(X_train, open('X_train', 'wb'))


def create_val_data(
    val_path,
    pca_path
):
    pca = pickle.load(open(pca_path, 'rb'))
    rew_path = os.path.join(val_path, 'rewards.csv')
    rewards = pandas.read_csv(rew_path, header=None).values[:, 1]
    X_val = []
    for root, _, img_files in os.walk(val_path):
        img_list = []
        for img_file in sorted(img_files):
            if img_file.endswith('png'):
                img_path = os.path.join(root, img_file)
                flat_image = process_image(img_path)
                img_list.append(flat_image)
        if len(img_list) == 0:
            continue

        rew_idx = int(root.split('/')[-1])

        row = np.array(img_list)
        row = pca.transform(row)
        row = row.reshape(250)

        row = np.concatenate([row, [rewards[rew_idx]]], axis=0)
        X_val.append(row)
    X_val = np.array(X_val, dtype='float32')
    pickle.dump(X_val, open('data_val', 'wb'))


def f1(y_true, y_pred):
    def recall(y_true, y_pred):
        """Recall metric.

        Only computes a batch-wise average of recall.

        Computes the recall, a metric for multi-label classification of
        how many relevant items are selected.
        """
        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
        recall = true_positives / (possible_positives + K.epsilon())
        return recall

    def precision(y_true, y_pred):
        """Precision metric.

        Only computes a batch-wise average of precision.

        Computes the precision, a metric for multi-label classification of
        how many selected items are relevant.
        """
        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
        precision = true_positives / (predicted_positives + K.epsilon())
        return precision
    precision = precision(y_true, y_pred)
    recall = recall(y_true, y_pred)
    return 2*((precision*recall)/(precision+recall+K.epsilon()))