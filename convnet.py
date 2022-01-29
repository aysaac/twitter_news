import tensorflow as tf
import wandb

from tf_utils import ResNet50
from wandb.keras import WandbCallback
print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))
#%%
sess = tf.compat.v1.Session(config=tf.compat.v1.ConfigProto(log_device_placement=True))
#%%
input_shape=(96,100)
model=ResNet50(input_shape=(64, 64, 3), classes=6)
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
# model.fit(X_train, Y_train,validation_data=(X_test, y_test), epochs = 10, batch_size = 32)
#%%


#%%

    # x=x+1
#%%
# wandb.init(project="feminism_analisis", entity="isaac_g",config=wandb_config)
