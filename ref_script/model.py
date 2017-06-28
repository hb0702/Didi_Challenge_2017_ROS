import numpy as np
import tensorflow as tf
import keras

from keras.models import Model
from keras.layers import Input
from keras.layers.pooling import MaxPooling2D
from keras.layers.normalization import BatchNormalization
from keras.layers.convolutional import Conv2D, Conv2DTranspose
from keras.layers.merge import Concatenate


def fcn_model(input_shape = (64,256,2), summary = True):
    
    input_img = Input(shape = input_shape)

    #normalized_input = Lambda(lambda z: z / 255. - .5)(input_img)
    # Todo: normalize two separate channel
    conv1 = Conv2D(64, (3, 3), activation='relu', kernel_initializer="glorot_uniform",
                          padding = 'same', name='conv1')(input_img)
    

    bn21 = BatchNormalization(name='bn21')(conv1)
    conv21 = Conv2D(64, (3, 3), activation='relu', kernel_initializer="glorot_uniform",
                          padding = 'same', name='conv21')(bn21)
    bn22 = BatchNormalization(name='bn22')(conv21)
    conv22 = Conv2D(64, (3, 3), activation='relu', kernel_initializer="glorot_uniform",
                          padding = 'same', name='conv22')(bn22)
    maxpool2 = MaxPooling2D((2,2), name='maxpool2')(conv22)
    
    
    bn31 = BatchNormalization(name='bn31')(maxpool2)
    conv31 = Conv2D(64, (3, 3), activation='relu', kernel_initializer="glorot_uniform",
                          padding = 'same', name='conv31')(bn31)
    bn32 = BatchNormalization(name='bn32')(conv31)
    conv32 = Conv2D(64, (3, 3), activation='relu', kernel_initializer="glorot_uniform",
                          padding = 'same', name='conv32')(bn32)
    maxpool3 = MaxPooling2D((2,2), name='maxpool3')(conv32)

    
    bn41 = BatchNormalization(name='bn41')(maxpool3)
    conv41 = Conv2D(64, (3, 3), activation='relu', kernel_initializer="glorot_uniform",
                          padding = 'same', name='conv41')(bn41)
    bn42 = BatchNormalization(name='bn42')(conv41)
    conv42 = Conv2D(64, (3, 3), padding = 'same', kernel_initializer="glorot_uniform", 
                    activation='relu', name='conv42')(bn42)
    maxpool4 = MaxPooling2D((2,2), name='maxpool4')(conv42)

    
    bn51 = BatchNormalization(name='bn51')(maxpool4)
    conv51 = Conv2D(64, (3, 3), activation='relu', kernel_initializer="glorot_uniform",
                          padding = 'same', name='conv51')(bn51)
    bn52 = BatchNormalization(name='bn52')(conv51)
    conv52 = Conv2D(64, (3, 3), padding = 'same', kernel_initializer="glorot_uniform", 
                    activation='relu', name='conv52')(bn52)
    maxpool5 = MaxPooling2D((2,2), name='maxpool5')(conv52)
    
    
    
    bn61 = BatchNormalization(name='bn61')(maxpool5)
    conv6 = Conv2D(64, (3, 3), padding="same", kernel_initializer="glorot_uniform", 
                    activation="relu", name="conv61")(bn61)
    bn62 = BatchNormalization(name='bn62')(conv6)
    deconv6 = Conv2DTranspose(64, (3, 3), kernel_initializer="glorot_uniform", name="deconv6", 
                    activation="relu", padding="same", strides=(2, 2))(bn62)

    
    concat7 = Concatenate(name='concat7')([conv51, deconv6])
    bn71 = BatchNormalization(name='bn71')(concat7)
    conv7 = Conv2D(64, (3, 3), padding="same", kernel_initializer="glorot_uniform", 
                    activation="relu", name="conv71")(bn71)
    bn72 = BatchNormalization(name='bn72')(conv7)
    deconv7 = Conv2DTranspose(64, (3, 3), kernel_initializer="glorot_uniform", name="deconv7", 
                    activation="relu", padding="same", strides=(2, 2))(bn72)
    
    
    concat8 = Concatenate(name='concat8')([conv41, deconv7])
    bn81 = BatchNormalization(name='bn81')(concat8)
    conv81 = Conv2D(64, (3, 3), padding="same", kernel_initializer="glorot_uniform", 
                    activation="relu", name="conv81")(bn81)
    bn82 = BatchNormalization(name='bn82')(conv81)
    deconv81 = Conv2DTranspose(64, (3, 3), kernel_initializer="glorot_uniform", name="deconv81", 
                    activation="relu", padding="same", strides=(2, 2))(bn82)
    deconv82 = Conv2DTranspose(64, (3, 3), kernel_initializer="glorot_uniform", name="deconv82", 
                    activation="relu", padding="same", strides=(2, 2))(bn82)

    concat91 = Concatenate(name='concat91')([conv31, deconv81])
    bn911 = BatchNormalization(name='bn911')(concat91)
    conv91 = Conv2D(64, (3, 3), padding="same", kernel_initializer="glorot_uniform", 
                    activation="relu", name="conv91")(bn911)
    bn912 = BatchNormalization(name='bn912')(conv91)
    
    deconv91 = Conv2DTranspose(64, (3, 3), kernel_initializer="glorot_uniform", name="deconv91", 
                    activation="relu", padding="same", strides=(2, 2))(bn912)

    concat92 = Concatenate(name='concat92')([conv31, deconv82])
    bn921 = BatchNormalization(name='bn921')(concat92)
    conv92 = Conv2D(64, (3, 3), padding="same", kernel_initializer="glorot_uniform", 
                    activation="relu", name="conv92")(bn921)
    bn922 = BatchNormalization(name='bn922')(conv92)
    
    deconv92 = Conv2DTranspose(64, (3, 3), kernel_initializer="glorot_uniform", name="deconv92", 
                    activation="relu", padding="same", strides=(2, 2))(bn922)
    
    
    concat101 = Concatenate(name='concat101')([conv21, deconv91])
    bn101 = BatchNormalization(name='bn101')(concat101)
    conv101 = Conv2D(64, (3, 3), padding="same", kernel_initializer="glorot_uniform", 
                    activation="relu", name="conv101")(bn101)
    out1 = Conv2D(1, (1, 1), padding="same", kernel_initializer="glorot_uniform", 
                    activation="sigmoid", name="out1")(conv101)
    
    
    concat102 = Concatenate(name='concat102')([conv21, deconv92])
    bn102 = BatchNormalization(name='bn102')(concat102)
    conv102 = Conv2D(64, (3, 3), padding="same", kernel_initializer="glorot_uniform", 
                    activation="relu", name="conv102")(bn102)
    out2 = Conv2D(7, (1, 1), padding="same", kernel_initializer="glorot_uniform", 
                    activation="linear", name="out2")(conv102)
    
    #out = [out1, out2]
    out = Concatenate(name='concat_out')([out1, out2])
    

    model = Model(inputs=input_img, outputs=out)

    if summary:
        model.summary()

    return model

def my_loss(y_true, y_pred):

    seg_true,reg_true = tf.split(y_true, [1, 7], 3)
    seg_pred,reg_pred = tf.split(y_pred, [1, 7], 3)

    #ratio = 20*h*w/tf.reduce_sum(seg_true)
    #weight1 = ((ratio-1)*seg_true + 1)/ratio

    seg_loss = -tf.reduce_mean(tf.multiply(seg_true,tf.log(seg_pred+1e-8)) + tf.multiply(1-seg_true,tf.log(1-seg_pred+1e-8)))
    #seg_loss = -tf.reduce_mean(
    #    tf.multiply(tf.multiply(seg_true,tf.log(seg_pred)) + tf.multiply(1-seg_true,tf.log(1-seg_pred)), weight1))

    diff = tf.reduce_mean(tf.squared_difference(reg_true, reg_pred), axis=3, keep_dims=True)
    reg_loss = tf.reduce_mean(tf.multiply(seg_true,diff))

    #total_loss = reg_loss
    #total_loss = seg_loss
    total_loss = seg_loss + reg_loss
    return total_loss 

if __name__ == '__main__':
	model = fcn_model()
