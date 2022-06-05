# -*- coding: utf-8 -*-
import sys
sys.path.append("../src")

import tensorflow as tf
from config import *

def convBlock(input_layer, conv_channels, kernel_size = (3, 3), 
              pool_stride = (2, 2), dropout_rate = 0.1, padding = 'same', 
              activation = 'relu'):        
    """ backbone convBlock used for downsampling containing 
        Convolution -> Activation -> BatchNormalization -> Dropout -> 
        SeparableConvolution -> Activation -> BatchNormalization -> MaxPool
        returns:
            layer_1: for skip connection 
            layer_6: output layer
    """
    layer_1 = tf.keras.layers.Conv2D(conv_channels, kernel_size, 
            activation = activation, padding = padding, 
            kernel_initializer = 'he_normal')(input_layer)
    layer_2 = tf.keras.layers.BatchNormalization()(layer_1)
    layer_3 = tf.keras.layers.Dropout(dropout_rate)(layer_2)
    layer_4 = tf.keras.layers.SeparableConv2D(conv_channels, kernel_size, 
            activation = activation, padding = padding, 
            kernel_initializer = 'he_normal')(layer_3)
    layer_5 = tf.keras.layers.BatchNormalization()(layer_4)
    layer_6 = tf.keras.layers.MaxPool2D(pool_stride)(layer_5)
    
    return layer_1, layer_6

def gapConvBlock(input_layer, conv_channels, kernel_size = (3, 3), 
                 pool_stride = (2, 2), dropout_rate = 0.1, padding = 'same', 
                 activation = 'relu'):        
    """ backbone convBlock used for downsampling containing 
        Convolution -> Activation -> BatchNormalization -> Dropout -> 
        SeparableConvolution -> Activation -> BatchNormalization -> AvgPool
        returns:
            layer_1: for skip connection 
            layer_6: output layer
    """
    layer_1 = tf.keras.layers.Conv2D(conv_channels, kernel_size, 
            activation = activation, padding = padding, 
            kernel_initializer = 'he_normal')(input_layer)
    layer_2 = tf.keras.layers.BatchNormalization()(layer_1)
    layer_3 = tf.keras.layers.Dropout(dropout_rate)(layer_2)
    layer_4 = tf.keras.layers.SeparableConv2D(conv_channels, kernel_size, 
            activation = activation, padding = padding, 
            kernel_initializer = 'he_normal')(layer_3)
    layer_5 = tf.keras.layers.BatchNormalization()(layer_4)
    layer_6 = tf.keras.layers.AveragePooling2D(pool_stride)(layer_5)

    return layer_1, layer_6

def terminalConvBlock(input_layer, conv_channels, kernel_size = (3, 3), 
                      dropout_rate = 0.1, padding = 'same', activation = 'relu'):
    """ backbone terminalConvBlock containing
        Convolution -> Activation -> BatchNormalization -> Dropout -> 
        SeparableConvolution -> Activation -> BatchNormalization
        returns:
            layer_6: output layer
    """
    layer_1 = tf.keras.layers.Conv2D(conv_channels, kernel_size, 
            activation = activation, padding = padding, 
            kernel_initializer = 'he_normal')(input_layer)
    layer_2 = tf.keras.layers.BatchNormalization()(layer_1)
    layer_3 = tf.keras.layers.Dropout(dropout_rate)(layer_2)
    layer_4 = tf.keras.layers.SeparableConv2D(conv_channels, kernel_size, 
            activation = activation, padding = padding, 
            kernel_initializer = 'he_normal')(layer_3)
    layer_5 = tf.keras.layers.BatchNormalization()(layer_4)
    
    return layer_5

def transposeConvBlock(input_layer, skip_layer, conv_channels, 
                       kernel_size = (3, 3), transpose_kernel_size = (2, 2), 
                       dropout_rate = 0.1, padding = 'same', activation = 'relu', 
                       transpose_strides = (2, 2)):
    """ backbone transposeConvBlock containing
        TransposeConvolution skip_layer concatenatation -> Convolution 
        -> Activation -> BatchNormalization -> Dropout -> 
        SeparableConvolution -> Activation -> BatchNormalization
        returns:
            layer_7: output layer
    """
    layer_1 = tf.keras.layers.Conv2DTranspose(conv_channels, 
            transpose_kernel_size, strides = transpose_strides, 
            padding = padding)(input_layer)
    layer_2 = tf.keras.layers.concatenate([layer_1, skip_layer], axis = 3)
    
    layer_3 = tf.keras.layers.Conv2D(conv_channels, kernel_size, 
            activation = activation, padding = padding, 
            kernel_initializer = 'he_normal')(layer_2)
    layer_4 = tf.keras.layers.BatchNormalization()(layer_3)
    layer_5 = tf.keras.layers.Dropout(dropout_rate)(layer_4)
    layer_6 = tf.keras.layers.SeparableConv2D(conv_channels, kernel_size, 
            activation = activation, padding = padding, 
            kernel_initializer = 'he_normal')(layer_5)
    layer_7 = tf.keras.layers.BatchNormalization()(layer_6)    
    
    return layer_7

def getEncoder(inputs):
    """ Feature extraction encoder creates downsampling network.
        Returns:
            S1, S2, S3, S4: skip connecction layer
            D1, D2, D3, D4: down sampling layer
            T1: Terminal bridge layer
            model: Encoder
    """
    S1, D1 = convBlock(inputs, 8)
    S2, D2 = convBlock(D1, 16)
    S3, D3 = convBlock(D2, 16)
    S4, D4 = gapConvBlock(D3, 32)
    T1 = terminalConvBlock(D4, 64)

    model = tf.keras.Model(inputs = inputs, outputs = T1, 
                name = "WaferSegClassNet-Encoder")
    return (S1, S2, S3, S4, T1, model)

def getDecoder(encoder, S1, S2, S3, S4):
    """ mask creation decoder creates binary output mask.
        Returns:
            model: Decoder
    """
    inputs = encoder.input
    T1 = encoder.output
    U1 = transposeConvBlock(T1, S4, 32)
    U2 = transposeConvBlock(U1, S3, 16)
    U3 = transposeConvBlock(U2, S2, 16)
    U4 = transposeConvBlock(U3, S1, 8)
    outputs = tf.keras.layers.Conv2D(SEG_NUM_CLASSES, (1, 1), activation = "sigmoid", name = "WaferSegClassNet-Segmentation")(U4)
    
    model = tf.keras.models.Model(inputs = [inputs], outputs = [outputs])
    return model

def addProjectionHead(inputs, encoder):
    """ creating contrastive model by adding gap and dense over encoder.
    """
    T1 = encoder.output
    gap = tf.keras.layers.GlobalAveragePooling2D()(T1)
    d1 = tf.keras.layers.Dense(128, activation = "relu")(gap)
    model = tf.keras.Model(inputs = inputs, outputs = d1, name="WaferSegClassNet-contrastive")
    return model

def getModel(encoder, decoder):
    """ Combining Encoder and Decoder network
    """
    inputs = encoder.input
    for layer in encoder.layers:
        layer.trainable = False

    T1 = encoder.output
    gap = tf.keras.layers.GlobalAveragePooling2D()(T1)
    d1 = tf.keras.layers.Dense(64, activation = "relu")(gap)
    d2 = tf.keras.layers.Dense(38, activation = "softmax", name = "WaferSegClassNet-Classification")(d1)

    model = tf.keras.models.Model(inputs = [inputs], outputs = [decoder.output, d2])    
    return model
