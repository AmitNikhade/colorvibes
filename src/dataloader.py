from tqdm import tqdm
import cv2, numpy as np, os
from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2 


def load_data(train_data, im_s):
    train_X = []
    for file in tqdm(os.listdir(train_data)):
        try:
            img = cv2.imread(train_data+file)
            img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
            img = cv2.resize(img, (im_s, im_s))
            train_X.append(img)

        except:
            pass

    train_X = np.array(train_X).astype('float32') / 255.
    return train_X

