# -*- coding: utf-8 -*-
import sys
sys.path.append("../src")

from tqdm import tqdm 
from config import *
from utils import *
import numpy as np
import logging
import cv2
import os

def main():
    """ Runs data processing scripts to turn raw data from (../raw) into
        cleaned data ready to be trained (saved in ../processed).
    """
    logger = logging.getLogger(__name__)
    logger.info('[Info] Generating Images, Labels and Masks from raw data')

    if not os.path.exists(IMAGES_DIR):
        os.makedirs(IMAGES_DIR)
    if not os.path.exists(LABELS_DIR):
        os.makedirs(LABELS_DIR)
    if not os.path.exists(MASKS_DIR):
        os.makedirs(MASKS_DIR)

    data = np.load(INPUT_FILE)

    for i in tqdm(range(len(data["arr_0"]))):
        inp_image = data["arr_0"][i].astype(np.uint8)
        inp_image = cv2.resize(inp_image, IMAGE_SIZE, interpolation = cv2.INTER_AREA)

        rgb_image = np.zeros(IMAGE_SIZE + (3, ), dtype=np.uint8)
        rgb_mask = np.zeros(IMAGE_SIZE + (1, ), dtype=np.uint8)

        image_colors = np.array([[255, 0, 255], [0, 255, 255], [255, 255, 0]])
        mask_colors = np.array([[0], [0], [255]])

        rgb_image = image_colors[inp_image]
        rgb_mask = mask_colors[inp_image]
        
        # Display the RGB image
        np.save(os.path.join(IMAGES_DIR, f"Image_{i}.npy"), rgb_image)
        np.save(os.path.join(LABELS_DIR, f"Image_{i}.npy"), data["arr_1"][i])
        np.save(os.path.join(MASKS_DIR, f"Image_{i}.npy"), rgb_mask)
    logger.info('[Info] Dataset Created Sucessfully')

if __name__ == '__main__':
    logging.basicConfig(level = logging.INFO, filename = os.path.join(LOG_DIR, 'app.log'), format='%(asctime)s - %(name)s - %(levelname)s - %(message)s', filemode='w')

    sys.stdout = LoggerWriter(logging.info)
    sys.stderr = LoggerWriter(logging.error)

    main()
