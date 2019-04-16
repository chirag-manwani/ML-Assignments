import imageio as io
import numpy as np
import os
import pickle
import queue
from PIL import Image
from sklearn.decomposition import PCA


# Preprocessing data utilities


def process_image(
    img_path
):
    left_crop = 7
    right_crop = -7
    top_crop = 31
    bottom_crop = -13

    img = np.array(Image.open(img_path).convert('L'))
    img = (img-128) / 128
    img = img[top_crop:bottom_crop, left_crop:right_crop]

    return img.flatten()


def episode_pca(
    train_path
):
    for root, dir, img_files in os.walk(train_path):
        img_list = []
        for img_file in sorted(img_files):
            if img_file.endswith('png'):
                img_path = os.path.join(root, img_file)
                flat_image = process_image(img_path)
                img_list.append(flat_image)
        if len(img_list) == 0:
            continue
        X = np.array(img_list)
        pca = PCA(n_components=50)
        X = pca.fit_transform(X)
        pickle.dump(X, open(os.path.join(root, 'X'), 'wb'))


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


train_path = '../temp/A4/test'
episode_pca(train_path)
# episode_pca(train_path)
