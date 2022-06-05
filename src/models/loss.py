# -*- coding: utf-8 -*-
import tensorflow_addons as tfa
import tensorflow as tf

class SupervisedContrastiveLoss(tf.keras.losses.Loss):
    def __init__(self, temperature = 1, name = None):
        super(SupervisedContrastiveLoss, self).__init__(name = name)
        self.temperature = temperature

    def __call__(self, labels, feature_vectors, sample_weight=None):
        # Normalize feature vectors
        feature_vectors_normalized = tf.math.l2_normalize(feature_vectors, axis=1)
        # Compute logits
        logits = tf.divide(
            tf.matmul(
                feature_vectors_normalized, tf.transpose(feature_vectors_normalized)
            ),
            self.temperature,
        )
        return tfa.losses.npairs_loss(tf.squeeze(labels), logits)

def diceCoef(y_true, y_pred, smooth=tf.keras.backend.epsilon()):   
    y_true_f = tf.keras.backend.flatten(y_true)    
    y_pred_f = tf.keras.backend.flatten(y_pred)    
    intersection = tf.keras.backend.sum(y_true_f * y_pred_f)    
    return (2. * intersection + smooth) / (tf.keras.backend.sum(y_true_f * y_true_f) + tf.keras.backend.sum(y_pred_f * y_pred_f) + smooth)

def diceCoefLoss(y_true, y_pred):
    return 1.0 - diceCoef(y_true, y_pred)

def bceDiceLoss(y_true, y_pred):
    loss = tf.keras.losses.binary_crossentropy(y_true, y_pred) + diceCoefLoss(y_true, y_pred)
    return loss
