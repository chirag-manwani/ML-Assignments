import sys
import numpy as np
import utils


def main(
    args
):
    train_path = args[0]
    pca_path = args[1]

    utils.pca_fit(train_path, pca_path)
    utils.create_episode_data(train_path, pca_path)


if __name__ == "__main__":
    args = sys.argv
    main(args[1:])

# '/media/cmkmanwani/hdd/chirag/test'