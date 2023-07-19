#!/usr/bin/env python
# coding: utf-8

# In[22]:


import pickle
from keras.preprocessing import image
import numpy as np


# #  Provide_path_of_image_in_next_line

# In[31]:


img_path = r'C:\Users\HP 840\Desktop\foody\BagelSandwich\B020411XX_11162.jpg'


# In[33]:


pickled_model = pickle.load(open(r'C:\Users\HP 840\Desktop\food_class\foodcm.pkl','rb'))
class_names = ['ApplePie', 'BagelSandwich', 'Bibimbop', 'Bread', 'FriedRice', 'Pork']
img = image.load_img(img_path, target_size=(100, 100))
image_array = image.img_to_array(img)
image_array = image.img_to_array(img)
x_train = np.expand_dims(image_array, axis=0)

pred = pickled_model.predict(x_train)
pred = np.argmax(pred, axis=-1)
print(class_names[pred[0]])


