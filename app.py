from __future__ import division, print_function
# coding=utf-8
import sys
import os
import glob
import re
import numpy as np
# import tensorflow as tf
from matplotlib import pyplot
# import cv2
from skimage.color import rgb2gray, gray2rgb, rgb2lab, lab2rgb
import matplotlib.pyplot as plt

# Keras
from keras.applications.imagenet_utils import preprocess_input, decode_predictions
from keras.models import load_model
from keras.preprocessing import image

# Flask utils
from flask import Flask, redirect, url_for, request, render_template
from werkzeug.utils import secure_filename
from gevent.pywsgi import WSGIServer
from skimage.transform import resize

# Define a flask app

app = Flask(__name__)
app.config['SEND_FILE_MAX_AGE_DEFAULT'] = 1

@app.after_request
def add_header(response):
    # response.cache_control.no_store = True
    if 'Cache-Control' not in response.headers:
        response.headers['Cache-Control'] = 'no-store'
    return response

MODEL_PATH = 'static/model/Model.h5'

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
# model._make_predict_function()          # Necessary
# print('Model loaded. Start serving...')

# You can also use pretrained model from Keras
# Check https://keras.io/applications/
#from keras.applications.resnet50 import ResNet50
#model = ResNet50(weights='imagenet')
#model.save('')
print('Model loaded. Check http://127.0.0.1:5000/')


    
    
def model_predict(img_path, model, filename):
    # if os.path.exists("static/images/images/"+filename) is True:
    #     os.remove("static/images/images/"+filename)
    # else:
    #     pass
    
    # img = cv2.imread(img_path)
    # if os.path.exists("static/images/img2.png") is True:
    #     os.remove("static/images/img2.png")
    # img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
    im = Image.open(TestImagePath+file) #These two lines
    b, g, r = im.split()
    im = Image.merge("RGB", (r, g, b))
    img = im.resize((256,256))
    img = [np.array(img)]
    # img = [cv2.resize(img, (256,256))]
    test = np.array(img).astype('float32') / 255.

    im = gray2rgb(rgb2gray(test))
    im_embed = inception_embedding(im)
    im = rgb2lab(im)[:,:,:,0]
    im = im.reshape(im.shape+(1,))

    pred = model.predict([im, im_embed])
    pred = pred * 128

    decodings = np.zeros((len(pred),256, 256, 3))
  
    for i in range(len(pred)):
        pp = np.zeros((256, 256, 3))
        pp[:,:,0] = im[i][:,:,0]
        pp[:,:,1:] = pred[i]
        decodings[i] = lab2rgb(pp)
        
    if os.path.exists("static/uploads/"+filename) is True:
        os.remove("static/uploads/"+filename)
        
    if os.path.exists("static/images/img2.png") is True:
        os.remove("static/images/img2.png")
        if os.path.exists("static/images/img2.png") is False:
            pyplot.imsave("static/images/img2.png", lab2rgb(pp))
    else:
        pyplot.imsave("static/images/img2.png", lab2rgb(pp))
        # pass
    
    
    return "hello"

@app.route('/', methods=['GET'])
def index():
    # Main page
    # if os.path.exists("static/images/img2.png") is True:
    #     os.remove("static/images/img2.png")
    return render_template('index.html')


@app.route('/predict', methods=['GET', 'POST'])
def upload():
    if request.method == 'POST':
        if os.path.exists("static/images/img2.png") is True:
            os.remove("static/images/img2.png")
        # Get the file from post request
        f = request.files['file']

        # Save the file to ./uploads
        basepath = os.path.dirname(__file__)
        file_path = os.path.join(
            basepath, 'static/uploads', secure_filename(f.filename))
        print("############################################################",f.filename)
        try:
            f.save(file_path)
        except Exception as ex:
            print("no file")
        

        # Make prediction
        result = model_predict(file_path, model, f.filename)
        
        
        

        # Process your result for human
        # pred_class = preds.argmax(axis=-1)            # Simple argmax
        # pred_class = decode_predictions(preds, top=1)   # ImageNet Decode
        # result = str(pred_class[0][0][1])  
        # Convert to string
        
        return result
        
    return None
    

if __name__ == '__main__':
    app.config['TEMPLATES_AUTO_RELOAD'] = True
    app.config['SEND_FILE_MAX_AGE_DEFAULT'] = 0
    app.run(debug=True, use_reloader=True)
    
