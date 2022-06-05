# -*- coding: utf-8 -*-
import sys
sys.path.append("../src")

from sklearn.model_selection import train_test_split
import tensorflow as tf
from config import *
from utils import *
import numpy as np
import os

class waferSegDataLoaderContrastive(tf.keras.utils.Sequence):
    """Dataloader class to iterate over the data (as Numpy arrays) for
       Pre-Training encoder with npair Contrastive. 
    """
    def __init__(self, batchSize, imgSize, inputImgPaths, labelsImgPaths):
        self.batchSize = batchSize
        self.imgSize = imgSize
        self.inputImgPaths = inputImgPaths
        self.labelsImgPaths = labelsImgPaths

    def __len__(self):
        return len(self.inputImgPaths) // self.batchSize

    def __getitem__(self, idx):
        """Returns tuple (input, target) correspond to batch #idx."""
        i = idx * self.batchSize
        batchInputImgPaths = self.inputImgPaths[i : i + self.batchSize]
        batchLabelsImgPaths = self.labelsImgPaths[i : i + self.batchSize]

        x = np.zeros((self.batchSize,) + self.imgSize + (3,), dtype = "float32")
        z = np.zeros((self.batchSize,), dtype = "float32")
        for j, (input_image, input_label) in enumerate(zip(batchInputImgPaths, batchLabelsImgPaths)):
            img = np.load(input_image)
            x[j] = img.astype("float32") / 255.0
            
            lbl = np.load(input_label)
            lbl = CLASS_MAPPING[str(lbl)]
            z[j] = lbl

        return x, z

class waferSegClassDataLoader(tf.keras.utils.Sequence):
    """Dataloader class to iterate over the data (as Numpy arrays) for 
       segmentation and classification branch"""
    def __init__(self, batchSize, imgSize, inputImgPaths, targetImgPaths, 
                labelsImgPaths):
        self.batchSize = batchSize
        self.imgSize = imgSize
        self.inputImgPaths = inputImgPaths
        self.targetImgPaths = targetImgPaths
        self.labelsImgPaths = labelsImgPaths

    def __len__(self):
        return len(self.targetImgPaths) // self.batchSize

    def __getitem__(self, idx):
        """Returns tuple (input, (target1, target2)) correspond to batch #idx."""
        i = idx * self.batchSize
        batchInputImgPaths = self.inputImgPaths[i : i + self.batchSize]
        batchTargetImgPaths = self.targetImgPaths[i : i + self.batchSize]
        batchLabelsImgPaths = self.labelsImgPaths[i : i + self.batchSize]

        x = np.zeros((self.batchSize,) + self.imgSize + (3,), dtype = "float32")
        y = np.zeros((self.batchSize,) + self.imgSize + (SEG_NUM_CLASSES,), dtype = "float32")
        z = np.zeros((self.batchSize,) + (38,), dtype = "float32")
        for j, (input_image, input_mask, input_label) in enumerate(zip(batchInputImgPaths, batchTargetImgPaths, batchLabelsImgPaths)):
            img = np.load(input_image)
            x[j] = img.astype("float32") / 255.0
            
            msk = np.load(input_mask)
            msk = msk / 255.0
            
            y[j] = msk.astype("float32")
            
            lbl = np.load(input_label)
            lbl = CLASS_MAPPING[str(lbl)]
            lbl = tf.keras.utils.to_categorical(lbl, num_classes = CLS_NUM_CLASSES)
            z[j] = lbl

        return x, (y, z)

def getDataLoader(batch_size):
    """ Create dataloader and return dataloader object which can be used with 
        model.fit
    """
    inputImgPaths = sorted([os.path.join(IMAGES_DIR, x) for x in os.listdir(IMAGES_DIR)])
    targetImgPaths = sorted([os.path.join(MASKS_DIR, x) for x in os.listdir(MASKS_DIR)])
    labelsImgPaths = sorted([os.path.join(LABELS_DIR, x) for x in os.listdir(LABELS_DIR)])

    labels = []
    for label in labelsImgPaths:
        lbl = np.load(label)
        labels.append(CLASS_MAPPING[str(lbl)])

    X_train, X_test, y_train, y_test = train_test_split(inputImgPaths, labels, test_size = TEST_SIZE, random_state = SEED)

    trainInputImgPaths = X_train
    testInputImgPaths = X_test
    trainTargetImgPaths = [img.replace("Images", "Masks") for img in X_train]
    testTargetImgPaths = [img.replace("Images", "Masks") for img in X_test]
    trainLabelsImgPaths = [img.replace("Images", "Labels") for img in X_train]
    testLabelsImgPaths = [img.replace("Images", "Labels") for img in X_test]

    trainGenContrastive = waferSegDataLoaderContrastive(batchSize = batch_size, imgSize = IMAGE_SIZE, inputImgPaths = trainInputImgPaths, labelsImgPaths = trainLabelsImgPaths)
    testGenContrastive = waferSegDataLoaderContrastive(batchSize = batch_size, imgSize = IMAGE_SIZE, inputImgPaths = testInputImgPaths, labelsImgPaths = testLabelsImgPaths)

    trainGen = waferSegClassDataLoader(batchSize = batch_size, imgSize = IMAGE_SIZE, inputImgPaths = trainInputImgPaths, targetImgPaths = trainTargetImgPaths, labelsImgPaths = trainLabelsImgPaths)
    testGen = waferSegClassDataLoader(batchSize = batch_size, imgSize = IMAGE_SIZE, inputImgPaths = testInputImgPaths, targetImgPaths = testTargetImgPaths, labelsImgPaths = testLabelsImgPaths)

    return (trainGenContrastive, testGenContrastive), (trainGen, testGen)