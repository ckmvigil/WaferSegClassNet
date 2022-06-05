# -*- coding: utf-8 -*-
import sys
sys.path.append("../src")

from tqdm import tqdm 
from config import *
from utils import *
import numpy as np
import logging
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
        np.save(os.path.join(IMAGES_DIR, f"Image_{i}.npy"), data["arr_0"][i])
        np.save(os.path.join(LABELS_DIR, f"Image_{i}.npy"), data["arr_1"][i])
        mask = createMask(data["arr_0"][i])
        np.save(os.path.join(MASKS_DIR, f"Image_{i}.npy"), mask)
    logger.info('[Info] Dataset Created Sucessfully')

if __name__ == '__main__':
    logging.basicConfig(level = logging.INFO, filename = os.path.join(LOG_DIR, 'app.log'), format='%(asctime)s - %(name)s - %(levelname)s - %(message)s', filemode='w')

    sys.stdout = LoggerWriter(logging.info)
    sys.stderr = LoggerWriter(logging.error)

    main()
