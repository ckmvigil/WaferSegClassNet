# -*- coding: utf-8 -*-
import sys
sys.path.append("../src")
sys.path.append("../src/data/")

from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt
import tensorflow as tf
from dataloader import *
from tqdm import tqdm
from config import *
from utils import *
from loss import *
import numpy as np
import argparse
import logging
import os

def main():
    """ testing given image. 
    """
    (trainGenContrastive, testGenContrastive), (trainGen, testGen) = getDataLoader(batch_size=1)
    model = tf.keras.models.load_model(os.path.join(WEIGHTS_DIR, 'last.h5'), custom_objects={"diceCoef":diceCoef, "bce_dice_loss":bceDiceLoss})
    y_true_seg, y_pred_seg, y_true_cls, y_pred_cls = [], [], [], []
    with tqdm(total = int(38015*TEST_SIZE)) as pbar:
        for data in testGen:
            img, (mask, label) = data
            seg, cls = model.predict(img)
            y_true_seg.append(mask[0][:, :, 0] > 0.5)
            y_pred_seg.append(seg[0][:, :, 0] > 0.5)
            y_true_cls.append(CLASS_NAME_MAPPING[np.argmax(label[0], axis=-1)])
            y_pred_cls.append(CLASS_NAME_MAPPING[np.argmax(cls[0], axis=-1)])
            pbar.update(1)

    y_true_seg = np.array(y_true_seg)
    y_pred_seg = np.array(y_pred_seg)
    y_true_cls = np.array(y_true_cls)
    y_pred_cls = np.array(y_pred_cls)

    np.save("y_true_seg.npy", y_true_seg)
    np.save("y_pred_seg.npy", y_pred_seg)
    np.save("y_true_cls.npy", y_true_cls)
    np.save("y_pred_cls.npy", y_pred_cls)

    IOU_Score = []

    for i in tqdm(range(y_true_seg.shape[0])):
        m = tf.keras.metrics.MeanIoU(num_classes = 2)
        m.update_state(y_true_seg[i].flatten(), y_pred_seg[i].flatten())
        IOU_Score.append(m.result().numpy())
    
    logging.info("[Info] IOU_Score: ")
    logging.info(sum(IOU_Score) / len(IOU_Score))

    logging.info("[Info] DICE_Score: ")
    logging.info(diceCoef(y_true_seg, y_pred_seg))

    logging.info("[Info] Classification_report: ")
    logging.info(classification_report(y_true_cls, y_pred_cls))

    logging.info("[Info] ROC_AUC Curve: ")
    fpr, tpr, thresholds = roc_curve(y_true_cls.ravel(), y_pred_cls.ravel())
    auc = auc(fpr, tpr)

    fig, ax = plt.subplots(1,1)
    ax.plot(fpr, tpr, label='ROC curve WSCN (area = %0.4f)' % auc)
    ax.plot([0, 1], [0, 1], 'k--')
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.set_title('Receiver Operating Characteristic')
    ax.legend(loc="lower right")

    plt.savefig(os.path.join(INFERENCE_DIR, "ROC_Curve_WSCN.pdf"))

if __name__ == '__main__':
    # logging.basicConfig(level = logging.INFO, filename = os.path.join(LOG_DIR, 'app.log'), format='%(asctime)s - %(name)s - %(levelname)s - %(message)s', filemode='w')

    # sys.stdout = LoggerWriter(logging.info)
    # sys.stderr = LoggerWriter(logging.error)

    main()