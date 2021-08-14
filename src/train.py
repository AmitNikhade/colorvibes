import argparse
from tensorflow.keras.callbacks import ReduceLROnPlateau, TensorBoard, ModelCheckpoint, EarlyStopping
import model
from skimage.transform import resize
import numpy as np
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2 

import os
import logging
from skimage.color import rgb2gray, gray2rgb, rgb2lab
import dataloader
import datetime

logdir = "logs/scalars/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
tensorboard_callback = TensorBoard(log_dir=logdir)


def parse_args():
    desc = "Autoencoder"
    parser = argparse.ArgumentParser(description=desc)

    parser.add_argument('--epochs', type=int, default=1,
                        help='Number of epochs for training')

    parser.add_argument('--bs', type=int, default=64, help='batch size')

    parser.add_argument('--im_s', type=int, default=256, help='Image size')

    parser.add_argument('--train_data', type=str,
                        help='path_to_train_data', required=True)

    return parser.parse_args()


args = parse_args()

logger = logging.getLogger()

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
                        datefmt='%m/%d/%Y %H:%M:%S', level=logging.INFO)




# logger.setLevel(logging.DEBUG)


if args.train_data is None:
    logger.error("You need to provide the training data path")
# learning rate annealer
learning_rate_reduction = ReduceLROnPlateau(monitor='loss',
                                            patience=3,
                                            verbose=1,
                                            factor=0.5,
                                            min_lr=0.00001)

# # Saving Model
filepath = "model/Model_final.h5"
checkpoint = ModelCheckpoint(filepath,
                             save_best_only=True,
                             monitor='loss',
                             mode='min')
es = EarlyStopping(monitor='val_loss', mode='min', verbose=1)
# Callbacks function
model_callbacks = [learning_rate_reduction,
                   checkpoint, es, tensorboard_callback]
model = model.autoencoder()
model.compile(optimizer='adam', loss='mean_squared_error')
model.summary()


mn = MobileNetV2(weights='imagenet', include_top=True)



def inception_embedding(gray_rgb):
    def resize_gray(x):
        return resize(x, (224, 224, 3), mode='constant')
    rgb = np.array([resize_gray(x) for x in gray_rgb])
    rgb = preprocess_input(rgb)
    embed = mn.predict(rgb)
    return embed

BATCH_SIZE = args.bs
logger.info("processing data..")

train_X = dataloader.load_data(args.train_data, args.im_s)
logger.info("training started..")


datagen = ImageDataGenerator(
        shear_range=0.2,
        zoom_range=0.2,
        rotation_range=20,
        horizontal_flip=True)

def gen_train(dataset=train_X, batch_size = BATCH_SIZE):
    for batch in datagen.flow(dataset, batch_size = batch_size):
        X_batch = rgb2gray(batch)
        rgb = gray2rgb(X_batch)
        lab_batch = rgb2lab(batch)
        X_batch = lab_batch[:,:,:,0]
        X_batch = X_batch.reshape(X_batch.shape+(1,))
        Y_batch = lab_batch[:,:,:,1:] / 128
        logger.info("visit: amitnikhade.com")
        yield [X_batch, inception_embedding(rgb)], Y_batch


model.fit_generator(gen_train(train_X,BATCH_SIZE),
                    epochs=1,
                    verbose=1,
                    steps_per_epoch=train_X.shape[0]/BATCH_SIZE,
                    # ,
                     callbacks=model_callbacks
                     
                    )
logger.info("model obtained successfully")
