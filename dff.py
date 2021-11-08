import os
import tensorflow as tf
from tensorflow import keras
from pathlib import Path
import tensorflow_hub as hub
# data_folder = Path("source_data/text_files/")

export_dir = Path("./models/resnet_model.h5")
# new_model = tf.lite.TFLiteConverter.from_keras_model(export_dir)
# new_model = keras.models.load_model(export_dir)
new_model = keras.models.load_model(export_dir,custom_objects={'KerasLayer':hub.KerasLayer})
print("Model loaded")

