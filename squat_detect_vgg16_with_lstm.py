# -*- coding: utf-8 -*-
"""
Created on Thu Apr 16 19:26:45 2020

@author: pappuyadav
"""

%tensorflow_version 1.x
from tensorflow.keras.applications.vgg16 import VGG16, preprocess_input
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.preprocessing import sequence
import os
import numpy as np
from keras.utils import np_utils
from sklearn.utils import shuffle
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Dropout,Flatten,GlobalAveragePooling2D,Input,LSTM,Embedding
import time
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt
from tensorflow.keras.callbacks import TensorBoard
from keras.preprocessing.image import ImageDataGenerator
from scipy import ndimage
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Model, Sequential


Name="Squat_Detection{}".format(int(time.time()))

tensorboard=TensorBoard(log_dir='logs/{}'.format(Name))

#Loading Training Data Set
image_data_list=[]
for file in os.listdir('/content/train'):
    full_path='/content/train/'+ str(file)
    print(full_path)
    image=load_img(full_path,target_size=(224,224))
    image=img_to_array(image)
    image=image.reshape(1,image.shape[0],image.shape[1],image.shape[2])
    image=preprocess_input(image)
    image_data_list.append(image)
    
image_data=np.array(image_data_list)
image_data=np.rollaxis(image_data,1,0)
image_data=image_data[0] 

#Defining number of classes
num_classes=2
num_samples=image_data.shape[0]
labels=np.ones(num_samples,dtype='int64')
labels[0:99]=0  # first 100 images are class'0'
labels[100:199]=1  # next 100 images are class'1'
names=['no_squat','squat']
#convert class labels to one-hot encoding
Y = np_utils.to_categorical(labels, num_classes)
#Shuffle the training data set
xtrain,ytrain=shuffle(image_data,Y,random_state=2)

#Loading Test Data Set
image_data_list1=[]
for file in os.listdir('/content/test'):
    print(file)
    full_path='/content/test/'+ str(file)
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
xtest,ytest=shuffle(image_data1,Y1,random_state=2)


#Using VGG16 as Feature Extraxtor and then LSTM as a classifier
img_input=Input(shape=(224,224,3))
model1=VGG16(input_tensor=img_input,include_top=False,weights='imagenet') #as feature extractor
last_layer=model1.get_layer('block5_pool').output
x=Flatten(name='flatten')(last_layer)
#x=Dense(128,name='squat1',activation='relu',trainable=True)(x)
squat_detect_model=Model(inputs=img_input,outputs=x)
print(squat_detect_model.summary())
t=time.time()
#Extracting features
vgg16_featuretrain=squat_detect_model.predict(xtrain)
vgg16_featuretest=squat_detect_model.predict(xtest)
print(vgg16_featuretrain.shape)
print(vgg16_featuretest.shape)
#Now we have extracted feature vectors using VGG16 pre-trained model. We will now train LSTM network
#..using these features and then make predictions of squat or no_squat.For this we will feed vgg16_feature
#to LSTM network and then train the model and make predictions

X_train=vgg16_featuretrain
Y_train=ytrain
X_test=vgg16_featuretest
Y_test=ytest
# Truncating and padding input sequences of data 
max_length = 50
X_train = sequence.pad_sequences(X_train, maxlen=max_length)
X_test = sequence.pad_sequences(X_test, maxlen=max_length)

#Building the LSTM architecture
size=50
embedding_vecor_length = 32
lstm_model = Sequential()
lstm_model.add(Embedding(size,embedding_vecor_length, input_length=max_length))
lstm_model.add(LSTM(100))
lstm_model.add(Dense(num_classes, activation='sigmoid'))
lstm_model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
print(lstm_model.summary())
hist=lstm_model.fit(X_train, Y_train, epochs=45, batch_size=10,verbose=1, validation_data=(X_test,Y_test), callbacks=[tensorboard])

print('Training time: %s' % (t - time.time()))
(loss, accuracy) = lstm_model.evaluate(X_test,Y_test, batch_size=10, verbose=1)
print("[INFO] loss={:.4f}, accuracy: {:.4f}%".format(loss,accuracy * 100))



temp=lstm_model.predict(X_test)
print('accuracy score is =',accuracy_score(ytest.argmax(axis=1),temp.argmax(axis=1))*100,'%')
print(confusion_matrix(ytest.argmax(axis=1), temp.argmax(axis=1)))
print(classification_report(ytest.argmax(axis=1),temp.argmax(axis=1)))

# visualizing losses and accuracy
train_loss=hist.history['loss']
val_loss=hist.history['val_loss']
train_acc=hist.history['acc']
val_acc=hist.history['val_acc']
xc=range(45)

plt.figure(1,figsize=(7,5))
plt.plot(xc,train_loss)
plt.plot(xc,val_loss)
plt.xlabel('num of Epochs')
plt.ylabel('loss')
plt.title('train_loss vs val_loss')
plt.grid(True)
plt.legend(['train','val'])
#print plt.style.available # use bmh, classic,ggplot for big pictures
plt.style.use(['classic'])

plt.figure(2,figsize=(7,5))
plt.plot(xc,train_acc)
plt.plot(xc,val_acc)
plt.xlabel('num of Epochs')
plt.ylabel('accuracy')
plt.title('train_acc vs val_acc')
plt.grid(True)
plt.legend(['train','val'],loc=4)
#print plt.style.available # use bmh, classic,ggplot for big pictures
plt.style.use(['classic'])



#Now saving the pretrained weights
# serialize model to JSON
model_json = lstm_model.to_json()
with open("trained_data_squats.json", "w") as json_file:
    json_file.write(model_json)
# serialize weights to HDF5
lstm_model.save_weights("trained_data_squats.h5")
print("Saved trained_data to disk")
