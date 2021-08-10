# from __future__ import division, print_function
# # coding=utf-8

# import os

# import numpy as np
# from PIL import Image

# # import tensorflow as tf
# from matplotlib import pyplot
# # import cv2
# from skimage.color import rgb2gray, gray2rgb, rgb2lab, lab2rgb


# # Keras
# # from keras.applications.imagenet_utils import preprocess_input
# from keras.models import load_model
# from keras.preprocessing import image

# # Flask utils
# from flask import Flask, request, render_template
# from werkzeug.utils import secure_filename
# from gevent.pywsgi import WSGIServer
# from skimage.transform import resize
# from tqdm import tqdm

# from keras.applications.mobilenet_v2 import MobileNetV2, preprocess_input

# MODEL_PATH = 'static/model/Model_final2.h5'
# model = load_model(MODEL_PATH)

# inception = MobileNetV2(weights='imagenet', include_top=True)

# def inception_embedding(gray_rgb):
#     def resize_gray(x):
# #         return resize(x, (299, 299, 3), mode='constant')
#         return resize(x, (224, 224, 3), mode='constant')
#     rgb = np.array([resize_gray(x) for x in gray_rgb])
#     rgb = preprocess_input(rgb)
#     embed = inception.predict(rgb)
#     return embed

# # Prediction on the test data
# TestImagePath="/home/lucifer/Downloads/"

# test = []
# t = 0
# for file in tqdm(os.listdir(TestImagePath)):
#     try:
#         t = t+1
# #         img = cv2.imread(TestImagePath+file)
# #         img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
#         im = Image.open("/home/lucifer/Downloads/ashim-d-silva-nUjujsVeiYo-unsplash.jpg") #These two lines
#         b, g, r = im.split()
#         im = Image.merge("RGB", (r, g, b))
#         img = im.resize((256,256))
#         img = np.array(img)
#         test.append(img)
#         if t == 1:
#             break
#     except:
#         pass

# from matplotlib import pyplot
# test = np.array(test).astype('float32') / 255.

# im = gray2rgb(rgb2gray(test))
# im_embed = inception_embedding(im)
# im = rgb2lab(im)[:,:,:,0]
# im = im.reshape(im.shape+(1,))

# pred = model.predict([im, im_embed])
# pred = pred * 128

# decodings = np.zeros((len(pred),256, 256, 3))

# for i in range(len(pred)):
#     pp = np.zeros((256, 256, 3))
#     pp[:,:,0] = im[i][:,:,0]
#     pp[:,:,1:] = pred[i]
#     decodings[i] = lab2rgb(pp)
# #     cv2.imwrite("img_"+str([i])+".jpg", lab2rgb(pp))
#     pyplot.imsave("img_5"+str([i])+".jpg", lab2rgb(pp))