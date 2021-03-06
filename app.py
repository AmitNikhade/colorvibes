from __future__ import division, print_function
# coding=utf-8

import os

import numpy as np
from PIL import Image

# import tensorflow as tf
from matplotlib import pyplot
# import cv2
from skimage.color import rgb2gray, gray2rgb, rgb2lab, lab2rgb


# tensorflow.Keras

# from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image

# Flask utils
from flask import Flask, request, render_template
from werkzeug.utils import secure_filename
from gevent.pywsgi import WSGIServer
from skimage.transform import resize

# Define a flask app

app = Flask(__name__)
app.config['SEND_FILE_MAX_AGE_DEFAULT'] = 0

@app.after_request
def add_header(r):
    """
    Add headers to both force latest IE rendering engine or Chrome Frame,
    and also to cache the rendered page for 10 minutes.
    """
    r.headers["Cache-Control"] = "no-cache, no-store, must-revalidate"
    r.headers["Pragma"] = "no-cache"
    r.headers["Expires"] = "0"
    r.headers['Cache-Control'] = 'public, max-age=0'
    return r

MODEL_PATH = 'static/model/Model_final2.h5'

# from tensorflow.keras.applications.nasnet import NASNetLarge
from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2, preprocess_input
inception = MobileNetV2(weights=None, include_top=True)
inception.load_weights('static/model/MNV2.h5')

def inception_embedding(gray_rgb):
    def resize_gray(x):
#         return resize(x, (299, 299, 3), mode='constant')
        return resize(x, (224, 224, 3), mode='constant')
    rgb = np.array([resize_gray(x) for x in gray_rgb])
    rgb = preprocess_input(rgb)
    embed = inception.predict(rgb)
    return embed

from tensorflow.keras.models import load_model
# Load your trained model
model = load_model(MODEL_PATH)

print('Model loaded. Check http://127.0.0.1:5000/')


    
    
# @app.route('/', methods = ['POST', 'GET'])
# @app.route('/model_predict', methods = ['POST', 'GET'])
def model_predict(img_path, model, filename):
    
    im = Image.open(img_path) #These two lines
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
        
    import time
    new_graph_name = "graph" + str(time.time()) + ".png"

    for filename in os.listdir('static/'):
        if filename.startswith('graph'):  # not to remove other images
            os.remove('static/' + filename)
    pyplot.imsave("static/"+new_graph_name, lab2rgb(pp))
    # if os.path.exists("static/uploads/"+filename) is True:
    #     os.remove("static/uploads/"+filename)
        
    # if os.path.exists("static/images/img2.png") is True:
    #     os.remove("static/images/img2.png")
    #     if os.path.exists("static/images/img2.png") is False:
    #         pyplot.imsave("static/images/img2.png", lab2rgb(pp))
    # else:
    #     os.mkdir()
    #     pyplot.imsave("static/images/img2.png", lab2rgb(pp))
    #     # pass
    
    return new_graph_name
    # return render_template('index.html', graph=new_graph_name)

@app.route('/', methods=['GET'])
def index():
    # Main page
    # if os.path.exists("static/images/img2.png") is True:
    #     os.remove("static/images/img2.png")
    return render_template('index.html')


@app.route('/predict', methods=['GET', 'POST'])
def upload():
    if request.method == 'POST':
        # if os.path.exists("static/images/img2.png") is True:
        #     os.remove("static/images/img2.png")
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
        # print(result)
        return render_template('index.html', result), result
        
    return None
    

if __name__ == '__main__':
    app.config['TEMPLATES_AUTO_RELOAD'] = True
    app.config['SEND_FILE_MAX_AGE_DEFAULT'] = 0
    app.run(debug=True, use_reloader=True)
    
