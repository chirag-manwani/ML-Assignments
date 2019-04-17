import imageio as io
import numpy as np
import os
import pickle
import queue
from PIL import Image
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

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
    train_path
):
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
    pickle.dump(pca, open(os.path.join('pca'), 'wb'))


def create_episode_data(
    episode_base_path,
    output_path,
    batch=7
):

    img_q = queue.Queue()
    X = None
    for i in range(batch):
        img = io.imread(img_paths[i])
        img = img[top_crop:bottom_crop, left_crop:right_crop]
        img = np.reshape(img, img.shape[0] * img.shape[1] * img.shape[2])
        img_q.put(img)

    for idx in range(num_imgs-batch+1):
        rows = fun(img_q, label)
        X = np.vstack(X, rows)
        break
