import os
from io import BytesIO
import numpy as np
import tensorflow as tf
from tensorflow import keras
import tensorflow_hub as hub
import json
from pathlib import Path
from PIL import Image 

def load_CNN_model():
    export_dir = Path('./models/resnet_model.h5')
    new_model = keras.models.load_model(export_dir,custom_objects={'KerasLayer':hub.KerasLayer})
    print("Model loaded")
    return new_model

def read_imagefile(file) -> Image.Image:
    image = Image.open(BytesIO(file))
    return image

def preprocess_image(image : Image.Image):
    input_size = (250,250)
    image = image.resize(input_size)
    image = np.asarray(image) # np.asarray as we don't want copy of that image
    image = image/255 # normalization
    return image 

def prediction_class_index():
    resnet_class_index = dict(json.load(open('class_index.json')))
    index_to_label = {v:k for k,v in resnet_class_index.items()}
    return index_to_label

def predict_image(image :np.ndarray):
    image = image.reshape(1,250,250,3)   # batch size = 1 
    model = load_CNN_model()
    outputs=model.predict(image) # predict method given by pre-trained CNN model
    y_hat = np.argmax(outputs)
    predicted_idx =y_hat.item()
    index_to_label = prediction_class_index()
    return index_to_label[predicted_idx]
    