#!/usr/bin/env python
# coding: utf-8

# # # Food Classification : Using Tensor-Flow & Keras

# In[64]:


# Import  Relevant Lib. 

import os
import PIL
import pickle
import numpy as np
import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
from tensorflow.keras import layers
from keras.preprocessing import image
from tensorflow.keras.models import Sequential
from keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.preprocessing import image_dataset_from_directory


# In[73]:


# Load training and validation image datasets
# Local Directory Path having images : in Sub-class folder format


train_dir = r'C:\Users\HP 840\Desktop\foody'
test_dir = r'C:\Users\HP 840\Desktop\foody'


train_data = tf.keras.utils.image_dataset_from_directory(
  train_dir,
  validation_split=0.2,
  subset="training",
  seed=220,
  image_size=(100, 100),
  batch_size=1)

test_data = tf.keras.utils.image_dataset_from_directory(
  test_dir,
  validation_split=0.2,
  subset="validation",
  seed=555,
  image_size=(100, 100),
  batch_size= 1)


# In[74]:


#enumerate food categories

class_names = train_data.class_names
print(class_names)


# In[75]:


# Data Normalisation , Prefetch & Caching dataSet


AUTOTUNE = tf.data.AUTOTUNE
normalization_layer = tf.keras.layers.Rescaling(1/255)
train_data = train_data.cache().prefetch(buffer_size=AUTOTUNE)
test_data = test_data.cache().prefetch(buffer_size=AUTOTUNE)


# In[76]:


# The CNN Struchure for M-Class 

num_classes = 6

model = Sequential([
  
  layers.Conv2D(32,(3,3), padding='same', activation='relu' , input_shape=(100,100,3)),
  layers.MaxPooling2D(),
  layers.Conv2D(32, 3, padding='same', activation='relu'),
  layers.MaxPooling2D(),
  layers.Conv2D(64, 3, padding='same', activation='relu'),
  layers.MaxPooling2D(),
  layers.Flatten(),
  layers.Dense(128, activation='relu'),
  layers.Dense(num_classes),
  ])


# In[77]:


#Model Compilation

model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])


# In[78]:


# Model Training on Image-Set

epochs=15
history = model.fit(
  train_data,
  validation_data=test_data,
  epochs=epochs
)


# # test your images here : by providing image-path in img_path from with correct file-format

# In[79]:


# Test any New-Image from given path (img_path)
  
img_path = r'C:\Users\HP 840\Desktop\a-test.jpg'
img = image.load_img(img_path, target_size=(100, 100))
image_array = image.img_to_array(img)
image_array = image.img_to_array(img)
x_train = np.expand_dims(image_array, axis=0)
pred = model.predict(x_train)
#print(pred)
# find the index of the class with maximum score
pred = np.argmax(pred, axis=-1)
# print the label of the class with maximum score
print(class_names[pred[0]])


# In[89]:


#pickle.dump(model, open( r'C:\Users\HP 840\Desktop\foodcm.pkl', 'wb'))



# In[ ]:




