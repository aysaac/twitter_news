import tensorflow as tf
from tensorflow.keras.layers import Input, Add, Dense, Activation, ZeroPadding2D, BatchNormalization, Flatten, Conv2D, AveragePooling2D, MaxPooling2D, GlobalMaxPooling2D
from tensorflow.keras.initializers import random_uniform, glorot_uniform
from tensorflow.keras.models import Model
print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))
#%%
# sess = tf.compat.v1.Session(config=tf.compat.v1.ConfigProto(log_device_placement=True))
#%%
input_shape=(96,100)

X_input = Input(input_shape)
X = ZeroPadding2D((3, 3))(X_input)
X = Conv2D(5,filters=6, kernel_size=1, strides=(2, 2), padding='valid', kernel_initializer=glorot_uniform(seed=0))(X)
X = BatchNormalization(axis=3)(X, training=True)
X = Activation('relu')(X)

