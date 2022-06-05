# -*- coding: utf-8 -*-
import sys
sys.path.append("../src")
sys.path.append("../src/data/")

import tensorflow as tf
from dataloader import *
from network import *
from loss import *
from config import *
from utils import *
import logging
import os

def main():
    """ training WaferSegClassNet model 
    """
    logger = logging.getLogger(__name__)
    logger.info("[Info] Getting DataLoader")
    (trainGenContrastive, testGenContrastive), (trainGen, testGen) = getDataLoader(BATCH_SIZE)
    logger.info("[Info] Creating Network")
    inputs = tf.keras.layers.Input(IMAGE_SIZE + (3,))    
    S1, S2, S3, S4, T1, encoder = getEncoder(inputs)
    model_contrastive = addProjectionHead(inputs, encoder)
    logger.info("[Info] Summary of contrastive model \n")
    logger.info(model_contrastive.summary())

    logger.info("[Info] Starting pretraining of encoder")
    model_contrastive.compile(optimizer = tf.keras.optimizers.Adam(INITIAL_LEARNING_RATE), loss = SupervisedContrastiveLoss(TEMPERATURE))
    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(os.path.join(WEIGHTS_DIR, 'WaferSegClassNet_Best_Contrastive.h5'), monitor = 'val_loss', verbose = 1, save_best_only = True, save_weights_only = False),
        tf.keras.callbacks.ReduceLROnPlateau(monitor = 'val_loss', factor = 0.1, patience = 20, min_lr = 0.00001)
    ]
    model_contrastive.fit(trainGenContrastive, validation_data = testGenContrastive, epochs = EPOCHS, callbacks = callbacks)  
    logger.info("[Info] Encoder Pre-Training is Finished")

    decoder = getDecoder(encoder, S1, S2, S3, S4)
    model = getModel(encoder, decoder)
    
    logger.info("[Info] Summary of model \n")
    logger.info(model.summary())

    model.compile(optimizer = tf.keras.optimizers.Adam(INITIAL_LEARNING_RATE), loss = {"WaferSegClassNet-Segmentation": bceDiceLoss, "WaferSegClassNet-Classification": "categorical_crossentropy"}, metrics = {"WaferSegClassNet-Segmentation": "accuracy", "WaferSegClassNet-Segmentation": diceCoef, "WaferSegClassNet-Classification": "accuracy"})
    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(os.path.join(WEIGHTS_DIR, 'WaferSegClassNet_Best.h5'), monitor = 'val_loss', verbose = 1, save_best_only = True, save_weights_only = False),
        tf.keras.callbacks.ReduceLROnPlateau(monitor = 'val_loss', factor = 0.1, patience = 20, min_lr = 0.00001)
    ]
    model.fit(trainGen, validation_data = testGen, epochs = EPOCHS // 2, verbose = 1, callbacks = callbacks)
    logger.info("[Info] Training Finished")

if __name__ == '__main__':
    logging.basicConfig(level = logging.INFO, filename = os.path.join(LOG_DIR, 'app.log'), format='%(asctime)s - %(name)s - %(levelname)s - %(message)s', filemode='w')

    sys.stdout = LoggerWriter(logging.info)
    sys.stderr = LoggerWriter(logging.error)

    main()
