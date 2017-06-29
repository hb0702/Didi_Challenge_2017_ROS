import numpy as np
#import tensorflow as tf
import keras

from keras.models import Model
from keras.layers import Input
from keras.layers.pooling import MaxPooling2D
from keras.layers.normalization import BatchNormalization
from keras.layers.convolutional import Conv2D, Conv2DTranspose
from keras.layers.merge import Concatenate
from keras.layers.core import Lambda



def fcn_model(input_shape = (16,320,2), summary = True):
    
    input_img = Input(shape = input_shape)

    #normalized_input = Lambda(lambda z: (z - mean_tensor)/std_tensor)(input_img)
    
    conv1 = Conv2D(64, (3, 3), activation='relu', kernel_initializer="glorot_uniform",
                          padding = 'same', name='conv1')(input_img)
    

    bn21 = BatchNormalization(name='bn21')(conv1)
    conv21 = Conv2D(64, (3, 3), activation='relu', kernel_initializer="glorot_uniform",
                          padding = 'same', name='conv21')(bn21)
    bn22 = BatchNormalization(name='bn22')(conv21)
    conv22 = Conv2D(64, (3, 3), activation='relu', kernel_initializer="glorot_uniform",
                          padding = 'same', name='conv22')(bn22)
    maxpool2 = MaxPooling2D((2,2), name='maxpool2')(conv22)
    
    
    bn30 = BatchNormalization(name='bn30')(maxpool2)
    conv30 = Conv2D(64, (3, 3), activation='relu', kernel_initializer="glorot_uniform",
                          padding = 'same', name='conv30')(bn30)
    bn31 = BatchNormalization(name='bn31')(conv30)
    conv31 = Conv2D(64, (3, 3), activation='relu', kernel_initializer="glorot_uniform",
                          padding = 'same', name='conv31')(bn31)
    bn32 = BatchNormalization(name='bn32')(conv31)
    conv32 = Conv2D(64, (3, 3), activation='relu', kernel_initializer="glorot_uniform",
                          padding = 'same', name='conv32')(bn32)
    maxpool3 = MaxPooling2D((2,2), name='maxpool3')(conv32)

    
    
    bn40 = BatchNormalization(name='bn40')(maxpool3)
    conv40 = Conv2D(64, (3, 3), activation='relu', kernel_initializer="glorot_uniform",
                          padding = 'same', name='conv40')(bn40)
    bn41 = BatchNormalization(name='bn41')(conv40)
    conv41 = Conv2D(64, (3, 3), activation='relu', kernel_initializer="glorot_uniform",
                          padding = 'same', name='conv41')(bn41)
    bn42 = BatchNormalization(name='bn42')(conv41)
    conv42 = Conv2D(64, (3, 3), padding = 'same', kernel_initializer="glorot_uniform", 
                    activation='relu', name='conv42')(bn42)
    maxpool4 = MaxPooling2D((2,2), name='maxpool4')(conv42)

    
    
    bn50 = BatchNormalization(name='bn50')(maxpool4)
    conv50 = Conv2D(64, (3, 3), activation='relu', kernel_initializer="glorot_uniform",
                          padding = 'same', name='conv50')(bn50)
    bn51 = BatchNormalization(name='bn51')(conv50)
    conv51 = Conv2D(64, (3, 3), activation='relu', kernel_initializer="glorot_uniform",
                          padding = 'same', name='conv51')(bn51)
    bn52 = BatchNormalization(name='bn52')(conv51)
    conv52 = Conv2D(64, (3, 3), padding = 'same', kernel_initializer="glorot_uniform", 
                    activation='relu', name='conv52')(bn52)
    maxpool5 = MaxPooling2D((2,2), name='maxpool5')(conv52)
    
    
    
    bn60 = BatchNormalization(name='bn60')(maxpool5)
    conv60 = Conv2D(64, (3, 3), padding="same", kernel_initializer="glorot_uniform", 
                    activation="relu", name="conv60")(bn60)
    bn61 = BatchNormalization(name='bn61')(bn60)
    conv61 = Conv2D(64, (3, 3), padding="same", kernel_initializer="glorot_uniform", 
                    activation="relu", name="conv61")(bn61)
    bn62 = BatchNormalization(name='bn62')(conv61)
    deconv6 = Conv2DTranspose(64, (3, 3), kernel_initializer="glorot_uniform", name="deconv6", 
                    activation="relu", padding="same", strides=(2, 2))(bn62)

    
    concat7 = Concatenate(name='concat7')([conv51, deconv6])
    bn70 = BatchNormalization(name='bn70')(concat7)
    conv70 = Conv2D(64, (3, 3), padding="same", kernel_initializer="glorot_uniform", 
                    activation="relu", name="conv70")(bn70)
    bn71 = BatchNormalization(name='bn71')(conv70)
    conv71 = Conv2D(64, (3, 3), padding="same", kernel_initializer="glorot_uniform", 
                    activation="relu", name="conv71")(bn71)
    bn72 = BatchNormalization(name='bn72')(conv71)
    deconv7 = Conv2DTranspose(64, (3, 3), kernel_initializer="glorot_uniform", name="deconv7", 
                    activation="relu", padding="same", strides=(2, 2))(bn72)
    
    
    concat8 = Concatenate(name='concat8')([conv41, deconv7])
    bn80 = BatchNormalization(name='bn80')(concat8)
    conv80 = Conv2D(64, (3, 3), padding="same", kernel_initializer="glorot_uniform", 
                    activation="relu", name="conv80")(bn80)
    bn81 = BatchNormalization(name='bn81')(conv80)
    conv81 = Conv2D(64, (3, 3), padding="same", kernel_initializer="glorot_uniform", 
                    activation="relu", name="conv81")(bn81)
    bn82 = BatchNormalization(name='bn82')(conv81)
    deconv81 = Conv2DTranspose(64, (3, 3), kernel_initializer="glorot_uniform", name="deconv81", 
                    activation="relu", padding="same", strides=(2, 2))(bn82)
    deconv82 = Conv2DTranspose(64, (3, 3), kernel_initializer="glorot_uniform", name="deconv82", 
                    activation="relu", padding="same", strides=(2, 2))(bn82)

    
    concat91 = Concatenate(name='concat91')([conv31, deconv81])
    bn910 = BatchNormalization(name='bn910')(concat91)
    conv910 = Conv2D(64, (3, 3), padding="same", kernel_initializer="glorot_uniform", 
                    activation="relu", name="conv910")(bn910)
    bn911 = BatchNormalization(name='bn911')(conv910)
    conv911 = Conv2D(64, (3, 3), padding="same", kernel_initializer="glorot_uniform", 
                    activation="relu", name="conv911")(bn911)
    bn912 = BatchNormalization(name='bn912')(conv911)
    
    deconv91 = Conv2DTranspose(64, (3, 3), kernel_initializer="glorot_uniform", name="deconv91", 
                    activation="relu", padding="same", strides=(2, 2))(bn912)

    
    concat92 = Concatenate(name='concat92')([conv31, deconv82])
    bn920 = BatchNormalization(name='bn920')(concat92)
    conv920 = Conv2D(64, (3, 3), padding="same", kernel_initializer="glorot_uniform", 
                    activation="relu", name="conv920")(bn920)
    bn921 = BatchNormalization(name='bn921')(conv920)
    conv921 = Conv2D(64, (3, 3), padding="same", kernel_initializer="glorot_uniform", 
                    activation="relu", name="conv921")(bn921)
    bn922 = BatchNormalization(name='bn922')(conv921)
    
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
    
    
    bn11 = BatchNormalization(name='bn11')(conv102)
    conv11 = Conv2D(64, (3, 3), padding="same", kernel_initializer="glorot_uniform", 
                    activation="relu", name="conv11")(bn11)
    maxpool11 = MaxPooling2D((2,2), name='maxpool11')(conv11)

    bn12 = BatchNormalization(name='bn12')(maxpool11)
    conv12 = Conv2D(64, (3, 3), padding="same", kernel_initializer="glorot_uniform", 
                    activation="relu", name="conv12")(bn12)

    bn13 = BatchNormalization(name='bn13')(conv12)
    deconv13 = Conv2DTranspose(64, (3, 3), kernel_initializer="glorot_uniform", name="deconv13", 
                    activation="relu", padding="same", strides=(2, 2))(bn13)

    bn14 = BatchNormalization(name='bn14')(deconv13)
    conv14 = Conv2D(64, (3, 3), padding="same", kernel_initializer="glorot_uniform", 
                    activation="relu", name="conv14")(bn14)                   
    out2 = Conv2D(7, (1, 1), padding="same", kernel_initializer="glorot_uniform", 
                    activation="linear", name="out2")(conv14)
    
    #out = [out1, out2]
    out = Concatenate(name='concat_out')([out1, out2])
    

    model = Model(inputs=input_img, outputs=out)

    if summary:
        model.summary()

    return model

if __name__ == '__main__':
	model = fcn_model()