# from __future__ import division, print_function
# coding=utf-8
# import sys
# import os
# import glob
# import re
# import numpy as np
from datetime import datetime
# # import tensorflow as tf
# from matplotlib import pyplot
# import cv2
# from skimage.color import rgb2gray, gray2rgb, rgb2lab, lab2rgb
# import matplotlib.pyplot as plt

# Keras
# from tensorflow.keras.applications.imagenet_utils import preprocess_input, decode_predictions
# from tensorflow.keras.models import load_model
# from tensorflow.keras.preprocessing import image

# Flask utils
from flask import Flask, redirect, url_for, request, render_template
# from werkzeug.utils import secure_filename
# from gevent.pywsgi import WSGIServer
# from skimage.transform import resize

app = Flask(__name__)

@app.route('/')
def homepage():
    the_time = datetime.now().strftime("%A, %d %b %Y %l:%M %p")

    return """
    <h1>Hello heroku</h1>
    <p>It is currently {time}.</p>
    <img src="http://loremflickr.com/600/400" />
    """.format(time=the_time)

if __name__ == '__main__':
    app.run(debug=True, use_reloader=True)