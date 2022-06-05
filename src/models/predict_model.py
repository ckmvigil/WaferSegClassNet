# -*- coding: utf-8 -*-
# usage: python models/predict_model.py --image {path_of_image} 
import sys
sys.path.append("../src")

import matplotlib.pyplot as plt
import tensorflow as tf
from config import *
from utils import *
from loss import *
import numpy as np
import argparse
import logging
import os

def main(args):
    """ predicting given image. 
    """
    model = tf.keras.models.load_model(os.path.join(WEIGHTS_DIR, 'WaferSegClassNet_Best.h5'), custom_objects={"diceCoef":diceCoef, "bceDiceLoss":bceDiceLoss})
    img = np.load(args["image"]) / 255.0
    img = np.expand_dims(img, 0)
    seg, cls = model.predict(img)
    plt.imsave(os.path.join(INFERENCE_DIR, "{}.png".format(os.path.splitext(os.path.basename(args["image"]))[0])), seg[0][:, :, 0])
    logging.info("[Info] Mask image is saved in Inference directory")
    logging.info("[Info] your predicted class is {}".format(CLASS_NAME_MAPPING[np.argmax(cls[0], axis=-1)]))

if __name__ == '__main__':
    logging.basicConfig(level = logging.INFO, filename = os.path.join(LOG_DIR, 'app.log'), format='%(asctime)s - %(name)s - %(levelname)s - %(message)s', filemode='w')

    sys.stdout = LoggerWriter(logging.info)
    sys.stderr = LoggerWriter(logging.error)

    ap = argparse.ArgumentParser()
    ap.add_argument("-i", "--image", required=True,
        help="path to input image in numpy format")
    args = vars(ap.parse_args())

    main(args)