# -*- coding: utf-8 -*-
"""
Created on Thu Apr 16 19:34:06 2020

@author: pappuyadav
"""
# Please use this section to test the model using your own test data sets i.e. image frames stored in "/content/newtest"
#...from your own video
# load json and create model
from tensorflow.keras.applications.vgg16 import VGG16, preprocess_input
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.preprocessing import sequence
import os
import numpy as np
from keras.utils import np_utils
from sklearn.utils import shuffle
from keras.models import model_from_json
import os
import numpy
import tensorflow as tf

json_file = open('/content/trained_data_squats.json', 'r')
loaded_model_json=json_file.read()
json_file.close()
loaded_model = tf.keras.models.model_from_json(loaded_model_json)

# load weights into new model
loaded_model.load_weights("/content/trained_data_squats.h5")
print("The model weights have been loaded")

# evaluate loaded model on test data
loaded_model.compile(loss='categorical_crossentropy', optimizer='rmsprop', metrics=['accuracy'])

#Loading Test Data Set from test video frames for evaluation of the pretrained model
#Please randomly select 16 image frames and put them in "/content/newtest". Choose..
#..such that first 7 images are no_squat image frames and remaining 9 image frames are of squats
#..For this you can renames images in serial order

#Loading Test Data Set
image_data_list1=[]
for file in os.listdir('/content/newtest'):
    print(file)
    full_path='/content/newtest/'+ str(file)
    print(full_path)
    image=load_img(full_path,target_size=(224,224))
    image_input1=image
    image=img_to_array(image)
    image=image.reshape(1,image.shape[0],image.shape[1],image.shape[2])
    image=preprocess_input(image)
    image_data_list1.append(image)

image_data1=np.array(image_data_list1)
image_data1=np.rollaxis(image_data1,1,0)
image_data1=image_data1[0] 

#Defining number of classes
num_classes1=2
num_samples1=image_data1.shape[0]
labels1=np.ones(num_samples1,dtype='int64')
labels1[0:6]=0  # first 7 images are class'0'
labels1[7:15]=1  # remianing images are class'1'
names1=['no_squat','squat']
#convert class labels to one-hot encoding
Y1 = np_utils.to_categorical(labels1, num_classes1)
#shuffling test dataset
X,Y=shuffle(image_data1,Y1,random_state=2)
(loss, accuracy) = loaded_model.evaluate(X_test,Y_test, batch_size=10, verbose=1)
print("[INFO] loss={:.4f}, accuracy: {:.4f}%".format(loss,accuracy * 100))