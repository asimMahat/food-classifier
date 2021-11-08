import os
from PIL import Image 
from io import BytesIO
import numpy as np
import tensorflow as tf
from tensorflow import keras
import json
# from tensorflow.keras.applications.imagenet_utils import decode_predictions
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import tensorflow_hub as hub


# from skimage.transform import rescale

from pathlib import Path

input_shape = (250,250)

def read_imagefile(file) -> Image.Image:
    image = Image.open(BytesIO(file))
    return image

# def preprocess(image : Image.Image):
#     image = image.resize(input_shape)
#     image = np.asfarray(image)
#     image = image / 127.5 - 1.0 
#     image = np.expand_dims(image,0)
#     print (image.shape)
    
#     return image 

def preprocess(image : Image.Image):
    image = image.resize(input_shape)
    image = np.asarray(image)
    image = image/255
    print(image.dtype)
    return image 

# def load_model():
#     model_input = (224,224,3)
#     model = tf.keras.applications.MobileNetV2(model_input)
#     return model

# _model = load_model()

def load_model():
    export_dir = Path('./models/resnet_model.h5')
    # new_model = tf.lite.TFLiteConverter.from_keras_model(export_dir)
    new_model = keras.models.load_model(export_dir,custom_objects={'KerasLayer':hub.KerasLayer})
    print("Model loaded")
    return new_model

# my_reloaded_model = tf.keras.models.load_model((path),custom_objects={'KerasLayer':hub.KerasLayer})

model = load_model()


#  def predict(image : np.ndarray):
#     # predictions = _model.predict(image)
#     # predictions = decode_predictions(predictions)[0][0][1]
#     return predictions


resnet_class_index = dict(json.load(open('class_index.json')))

index_2_label = {v:k for k,v in resnet_class_index.items()}


def predict(image :np.ndarray):
    image = image.reshape(1,250,250,3)
    outputs=model.predict(image)
    print(type(outputs))
    y_hat = np.argmax(outputs)
    predicted_idx =y_hat.item()
    print (predicted_idx)
    return index_2_label[predicted_idx]
    