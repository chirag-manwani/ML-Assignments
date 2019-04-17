import sys
import numpy as np
import utils


def main(
    args
):
    train_path = args[0]
    utils.pca_fit(train_path)


if __name__ == "__main__":
    args = sys.argv
    main(args[1:])

# '/media/cmkmanwani/hdd/chirag/test'