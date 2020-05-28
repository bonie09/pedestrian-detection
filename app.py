# Import libraries
import os
import sys
import random
import math
import re
import time
import numpy as np
import tensorflow as tf
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import imghdr
import cv2
import skimage.io
import logging
import argparse

# Import flask libraries
import hashlib
import json
from time import time
from urllib.parse import urlparse
from uuid import uuid4
import visualize
from visualize import display_images
from visualize import apply_mask_instances
from werkzeug.utils import secure_filename

import requests
from flask import Flask, request, jsonify, render_template
import pickle

from mrcnn.config import Config

# Root directory of the project
ROOT_DIR = os.getcwd()

# Import Mask RCNN
sys.path.append(ROOT_DIR)  # To find local version of the library
from mrcnn import utils
from mrcnn import visualize
from mrcnn.visualize import display_images
import mrcnn.model as modellib
from mrcnn.model import log

# Directories
MODEL_DIR = os.path.join(ROOT_DIR, "logs")      #training logs(weights)
IMAGE_DIR = os.path.join(ROOT_DIR, "images")
MASKED_DIR = os.path.join(IMAGE_DIR, "masked")  #masked images folder
UPLOAD_DIR = os.path.join(IMAGE_DIR, "upload")  #uploaded images folder

# Path to pre-trained weights
PEDESTRIAN_WEIGHTS_PATH = os.getcwd()

# Logging confg
logging.basicConfig(level=logging.DEBUG, filename="log", filemode="a+",
                format="%(asctime)-15s %(levelname)-8s %(message)s")

############################################################
#  Configurations
############################################################

class PedestrianConfig(Config):
    """Configuration for training on the toy  dataset.
    Derives from the base Config class and overrides some values.
    """
    # Give the configuration a recognizable name
    NAME = "pedestrian"

    # We use a GPU with 12GB memory, which can fit two images.
    # Adjust down if you use a smaller GPU.
    IMAGES_PER_GPU = 1

    # Number of classes (including background)
    NUM_CLASSES = 1 + 1  # Background + pedestrian

    # Number of training steps per epoch
    STEPS_PER_EPOCH = 100

    # Skip detections with < 90% confidence
    DETECTION_MIN_CONFIDENCE = 0.9

class InferenceConfig(PedestrianConfig):
            # Set batch size to 1 since we'll be running inference on
            # one image at a time. Batch size = GPU_COUNT * IMAGES_PER_GPU
            GPU_COUNT = 1
            IMAGES_PER_GPU = 1

config = InferenceConfig()
config.display()

class_names = ['BG', 'pedestrian']

# Create model object in inference mode.
model = modellib.MaskRCNN(mode="inference", model_dir=MODEL_DIR, config=config)
print(PEDESTRIAN_WEIGHTS_PATH)

# Load pre-trained weights
PEDESTRIAN_WEIGHTS_FILE = os.path.join(PEDESTRIAN_WEIGHTS_PATH, "mask_rcnn_pedestrian.h5")
model.load_weights(PEDESTRIAN_WEIGHTS_FILE, by_name=True)

model.keras_model._make_predict_function()

logging.info('Model and weight have been loaded.')

# def color_splash(image, mask):
#     """Apply color splash effect.
#     image: RGB image [height, width, 3]
#     mask: instance segmentation mask [height, width, instance count]

#     Returns result image.
#     """
#     # Make a grayscale copy of the image. The grayscale copy still
#     # has 3 RGB channels, though.
#     gray = skimage.color.gray2rgb(skimage.color.rgb2gray(image)) * 255
#     # Copy color pixels from the original color image where mask is set
#     if mask.shape[-1] > 0:
#         # We're treating all instances as one, so collapse the mask into one layer
#         mask = (np.sum(mask, -1, keepdims=True) >= 1)
#         splash = np.where(mask, image, gray).astype(np.uint8)
#     else:
#         splash = gray.astype(np.uint8)
#     return splash

def run_detect(filename):
    base_file_name = os.path.basename(filename)
    saved_file_name = os.path.join(MASKED_DIR, base_file_name)
    logging.info('Loading image: %s', base_file_name)

    # Convert png with alpha channel with shape[2] == 4 into shape[2] ==3 RGB images
    image = skimage.io.imread(filename)
    if len(image.shape) > 2 and image.shape[2] == 4:
        image = cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)
    logging.info('Image shape info: %s', image.shape)

    # Run detection
    results = model.detect([image], verbose=1)
    r= results[0]
    logging.info('Runing model.detect([image], verbose=1).')

    # Just apply mask then save images
    print_img = visualize.apply_mask_instances(image, r['rois'], r['masks'], r['class_ids'], class_names, r['scores'])
    skimage.io.imsave(saved_file_name,print_img)
    logging.info('Finished apply_mask_instances.')

    return True


# Instantiate the Node
app = Flask(__name__)

@app.route('/', methods=['GET'])
def index():
    # Main page
    return render_template('index.html')

@app.route('/predict', methods=['GET', 'POST'])
def upload():
    if request.method == 'POST':
        # Get the file from post request
        f = request.files['file']

        # Save the file to ./uploads
        basepath = os.path.dirname(__file__)
        print(basepath)
        file_path = os.path.join(
            basepath, UPLOAD_DIR, secure_filename(f.filename))
        print(file_path)
        f.save(file_path)

        r = run_detect(file_path)

        return "DONE! Image saved to masked folder"
    return None


if __name__ == '__main__':

    from argparse import ArgumentParser
    parser = ArgumentParser()
    parser.add_argument('-p', '--port', default=5000, type=int, help='port to listen on')
    args = parser.parse_args()
    port = args.port

    app.run(debug=True)
