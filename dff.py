# A sample script to load the model 
import os
import tensorflow as tf
from tensorflow import keras
from pathlib import Path
import tensorflow_hub as hub

export_dir = Path("./models/resnet_model.h5")
new_model = keras.models.load_model(export_dir,custom_objects={'KerasLayer':hub.KerasLayer})
print("Model loaded")

