import numpy as np
import sys
import os
from PIL import Image
from pathlib import Path
from model import get_model, get_model_exp

if __name__ == "__main__":
    test_path = Path(sys.argv[1])
    weights_path = Path(sys.argv[2])

    model = get_model_exp()
    if weights_path.is_file():
        model.load_weights(str(weights_path))
    else:
        print('Model not found')
        exit

    model.summary()
    dirs = sorted(os.listdir(test_path))

    predictions = open('predictions.csv', 'w')
    predictions.write('id,Prediction\n')
    for root in dirs:
        img_list = []
        img_files = sorted(os.listdir(os.path.join(test_path, root)))
        for img_file in sorted(img_files):
            if img_file.endswith('png'):
                img_path = os.path.join(test_path, root, img_file)
                img = np.array(Image.open(img_path).convert('L'))
                # Can add cropping code
                img_list.append(img)

        stacked_img = np.stack(img_list, axis=2)
        stacked_img = np.expand_dims(stacked_img, axis=0)
        pred = int(model.predict(stacked_img)[0][0])
        predictions.write(str(int(root)) + ',' + str(pred) + '\n')
    predictions.close()
