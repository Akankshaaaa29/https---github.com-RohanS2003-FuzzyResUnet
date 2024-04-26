# source https://naomi-fridman.medium.com/multi-class-image-segmentation-a5cc671e647a
# import system libs
import os
import time
import random
import pathlib
import itertools
from glob import glob


# import data handling tools

import numpy as np
import pandas as pd


import matplotlib.pyplot as plt



# import Deep learning Libraries
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import backend as K
from tensorflow.keras.models import Model, load_model, save_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import Adam, Adamax
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.layers import Input, Activation, BatchNormalization, Dropout, Lambda, Conv2D, Conv2DTranspose, MaxPooling2D, concatenate

import keras
import keras.backend as K
from keras.callbacks import CSVLogger
import tensorflow as tf
from tensorflow.keras.utils import plot_model
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from tensorflow.keras.models import *
from tensorflow.keras.layers import *
from tensorflow.keras.optimizers import *
from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, EarlyStopping, TensorBoard
from tensorflow.keras.layers.experimental import preprocessing

# # import system libs
import os
import time
import random
import pathlib
import itertools
from glob import glob


# import data handling tools

import numpy as np
import pandas as pd


import matplotlib.pyplot as plt



# import Deep learning Libraries
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import backend as K
from tensorflow.keras.models import Model, load_model, save_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import Adam, Adamax
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.layers import Input, Activation, BatchNormalization, Dropout, Lambda, Conv2D, Conv2DTranspose, MaxPooling2D, concatenate

import keras
import keras.backend as K
from keras.callbacks import CSVLogger
import tensorflow as tf
from tensorflow.keras.utils import plot_model
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from tensorflow.keras.models import *
from tensorflow.keras.layers import *
from tensorflow.keras.optimizers import *
from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, EarlyStopping, TensorBoard
from tensorflow.keras.layers.experimental import preprocessing

import os
import glob

import numpy as np
import tensorflow as tf
from tensorflow.keras.optimizers import Adam
from keras.models import *
from keras.layers import *
from tensorflow.keras.utils import plot_model
from matplotlib import pyplot as plt
import keras.backend as K

# Ignore Warnings
import warnings
warnings.filterwarnings("ignore")

from tensorflow.keras.layers import Input, Conv2D, Dropout, MaxPooling2D, UpSampling2D, concatenate
from tensorflow.keras.models import Model

from tensorflow.keras.layers import Input, Conv2D, BatchNormalization, Activation, MaxPool2D, UpSampling2D, Concatenate, Add, GlobalAveragePooling2D, Lambda, multiply
from tensorflow.keras.models import Model
import tensorflow as tf

def fuzzy_attention(conv):
    Af_i = Activation('sigmoid')(UpSampling2D(size=(2, 2))(conv))
    Ari = Lambda(lambda x: 1 - x)(Af_i)
    Si = GlobalAveragePooling2D()(conv)
    Ai = multiply([Af_i, Si])
    difficulty = Lambda(lambda x: 1 - tf.abs(x[0] - x[1]))([Af_i, Ari])
    Ai = Activation('sigmoid')(difficulty)
    p = UpSampling2D(size=(2, 2))(conv)
    Ri = multiply([p, Ai])
    return Ri

def resblock(X,f):
    X_copy = X
    X = Conv2D(f,kernel_size=(1,1),kernel_initializer='he_normal')(X)
    X = BatchNormalization()(X)
    X = Activation('leaky_relu')(X)

    X = Conv2D(f,kernel_size=(3,3),padding='same',kernel_initializer='he_normal')(X)
    X = BatchNormalization()(X)

    X_copy = Conv2D(f,kernel_size=(1,1),kernel_initializer='he_normal')(X_copy)
    X_copy = BatchNormalization()(X_copy)

    X = Add()([X,X_copy])
    X = Activation('leaky_relu')(X)

    return X

def upsample_concat(x, skip):
    x = UpSampling2D((2, 2))(x)
    if x.shape[1] != skip.shape[1] or x.shape[2] != skip.shape[2]:
        x = tf.image.resize(x, (skip.shape[1], skip.shape[2]))
    merge = Concatenate()([x, skip])
    return merge



def create_resunet_model(input_shape=(256,256,3)):
    X_input = Input(input_shape)

    conv_1 = Conv2D(16,3,activation='leaky_relu',padding='same',kernel_initializer='he_normal')(X_input)
    conv_1 = BatchNormalization()(conv_1)
    conv_1 = Conv2D(16,3,activation='leaky_relu',padding='same',kernel_initializer='he_normal')(conv_1)
    conv_1 = BatchNormalization()(conv_1)
    pool_1 = MaxPool2D((2,2))(conv_1)

    conv_2 = resblock(pool_1,32)
    pool_2 = MaxPool2D((2,2))(conv_2)

    conv_3 = resblock(pool_2,64)
    pool_3 = MaxPool2D((2,2))(conv_3)

    conv_4 = resblock(pool_3,128)
    pool_4 = MaxPool2D((2,2))(conv_4)

    conv_5 = resblock(pool_4,256)
    conv_5 = fuzzy_attention(conv_5)

    up_1 = upsample_concat(conv_5,conv_4)
    up_1 = resblock(up_1,128)

    up_2 = upsample_concat(up_1,conv_3)
    up_2 = resblock(up_2,64)

    up_3 = upsample_concat(up_2,conv_2)
    up_3 = resblock(up_3,32)

    up_4 = upsample_concat(up_3,conv_1)
    up_4 = resblock(up_4,16)

    out = Conv2D(2,(1,1),kernel_initializer='he_normal',padding='same',activation='sigmoid')(up_4)
    model = Model(X_input,out)

    return model

modelresunet = create_resunet_model()
#modelresunet.summary()




#model.compile(loss="categorical_crossentropy", optimizer=keras.optimizers.Adam(learning_rate=0.001), metrics = ['accuracy',tf.keras.metrics.MeanIoU(num_classes=4), dice_coef, precision, sensitivity, specificity, dice_coef_necrotic, dice_coef_edema ,dice_coef_enhancing] )


