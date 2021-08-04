from __future__ import division, print_function
# coding=utf-8
import sys
import os
import glob
import re
import numpy as np
import tensorflow as tf
from matplotlib import pyplot
import cv2
from skimage.color import rgb2gray, gray2rgb, rgb2lab, lab2rgb
import matplotlib.pyplot as plt

# Keras
from tensorflow.keras.applications.imagenet_utils import preprocess_input, decode_predictions
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image

# Flask utils
from flask import Flask, redirect, url_for, request, render_template
from werkzeug.utils import secure_filename
from gevent.pywsgi import WSGIServer
from skimage.transform import resize



MODEL_PATH = 'dist/model/Model.h5'

from tensorflow.keras.applications.nasnet import NASNetLarge
inception = NASNetLarge(weights='imagenet', include_top=True)

def inception_embedding(gray_rgb):
    def resize_gray(x):
#         return resize(x, (299, 299, 3), mode='constant')
        return resize(x, (331, 331, 3), mode='constant')
    rgb = np.array([resize_gray(x) for x in gray_rgb])
    rgb = preprocess_input(rgb)
    embed = inception.predict(rgb)
    return embed

# Load your trained model
model = load_model(MODEL_PATH)

img_path = "/home/lucifer/Downloads/ashim-d-silva-nUjujsVeiYo-unsplash.jpg"

def model_predict(img_path, model):
    img = cv2.imread(img_path)
    img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
    img = [cv2.resize(img, (256,256))]
    test = np.array(img).astype('float32') / 255.

    im = gray2rgb(rgb2gray(test))
    im_embed = inception_embedding(im)
    im = rgb2lab(im)[:,:,:,0]
    im = im.reshape(im.shape+(1,))

    pred = model.predict([im, im_embed])
    pred = pred * 128

    decodings = np.zeros((len(pred),256, 256, 3))
    pp = np.zeros((256, 256, 3))
    pp[:,:,0] = im[:,:,0]
    pp[:,:,1:] = pred
    decodings = lab2rgb(pp)
    pyplot.imsave("img_5.jpg", lab2rgb(pp))
model_predict(img_path, model)