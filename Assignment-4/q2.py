import numpy as np
import random
import os
import _pickle as cPickle
import gzip
import sys
from pathlib import Path
from model import get_model, get_model_exp
from keras.models import load_model


def unzip_pickle(filename):
    with gzip.open(filename, 'rb') as f:
        loaded_object = cPickle.load(f)
        return loaded_object


def gen_data(
    pickle_file_path
):
    pickle_data = unzip_pickle(pickle_file_path)
    zero_data = pickle_data[0]
    one_data = pickle_data[1]

    # Can add cropping code here

    batch_data = []
    for row in zero_data:
        batch_data.append((row, np.array([0])))
    for row in one_data:
        batch_data.append((row, np.array([1])))

    np.random.shuffle(batch_data)

    X_batch = []
    Y_batch = []
    for x, y in batch_data:
        X_batch.append(x)
        Y_batch.append(y)
    return np.array(X_batch)/255, np.array(Y_batch)


if __name__ == "__main__":
    train_path = sys.argv[1]
    val_path = sys.argv[2]
    weights = Path(sys.argv[3])

    model = get_model_exp()
    try:
        model.load_weights(weights)
        print('Old Weights')
    except:
        print('New Weights')
    model.summary()

    train_pickle_files = os.listdir(train_path)
    X_val, Y_val = gen_data(val_path)

    num_epochs = 100
    max_f1 = 0
    for epoch in range(num_epochs):
        pickle_file_train = random.choice(train_pickle_files)

        X_train, Y_train = gen_data(
                os.path.join(train_path, pickle_file_train))
        hist = model.fit(X_train, Y_train, batch_size=32, epochs=epoch+1,
                         initial_epoch=epoch, validation_data=(X_val, Y_val))
        curr_f1 = hist.history['val_f1'][0]
        if curr_f1 > max_f1:
            model.save_weights('./models/w_' + str(curr_f1))
            max_f1 = curr_f1
    model.save_weights(weights)
    print(max_f1)
