import imageio as io
import numpy as np
import os
import pickle
import queue
from PIL import Image
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from pathlib import Path

# Preprocessing data utilities


def process_image(
    img_path
):
    left_crop = 6
    right_crop = -6
    top_crop = 40
    bottom_crop = -40

    img = np.array(Image.open(img_path).convert('L'))
    img = img[top_crop:bottom_crop, left_crop:right_crop]

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
    X = np.array(img_list)
    # scaler = StandardScaler()
    # X = scaler.fit_transform(X)
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
        X_file = Path(os.path.join(root, 'X_scaled'))
        print(X_file)
        if X_file.is_file():
            continue
        print(root)
        for img_file in sorted(img_files):
            if img_file.endswith('png'):
                img_path = os.path.join(root, img_file)
                flat_image = process_image(img_path)
                img_list.append(flat_image)
        X = np.array(img_list)
        X = pca.transform(X)
        root = Path(root)
        pickle.dump(X, open(root.joinpath(X_file), 'wb'))
