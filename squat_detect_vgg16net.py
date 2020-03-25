from tensorflow.keras.applications.vgg16 import VGG16, preprocess_input
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import os
import numpy as np
from keras.utils import np_utils
from sklearn.utils import shuffle
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense
import time
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt
from tensorflow.keras.callbacks import TensorBoard
from keras.preprocessing.image import ImageDataGenerator
from scipy import ndimage
from sklearn.model_selection import train_test_split

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
labels[100:189]=1  # next 90 images are class'1'
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



#Using VGG16 as Feature Extraxtor and then as a classifier
model1=VGG16(include_top=False,weights='imagenet') #as feature extractor
model2 = VGG16(include_top=True,weights='imagenet')  #as a classifier

#since we have only 2 classes and not 1000 classes, we replace the last layer..
#..by our own Dense layer named 'prediction' with 2 classes

last_layer=model2.get_layer('flatten').output
out1=Dense(4096,name='squat1',activation='relu',trainable=True)(last_layer)
out2=Dense(4096,name='squat2',activation='relu',trainable=True)(out1)
out3=Dense(num_classes,name='predict',activation='softmax',trainable=True)(out2)
#out=Dense(num_classes,activation='softmax')(last_layer)
squat_detect_model=Model(inputs=model2.input,outputs=out3)

#Freezing all the layers except the last layer.The last yer is the only trainable layer
#for layer in squat_detect_model.layers[:-1]:
   # layer.trainable=False


squat_detect_model.compile(loss='categorical_crossentropy',optimizer='adadelta',metrics=['accuracy'])
t=time.time()
trained_data=squat_detect_model.fit(xtrain,ytrain, batch_size=5, epochs=5, verbose=1, validation_data=(xtest,ytest), callbacks=[tensorboard])
print('Training time: %s' % (t - time.time()))
(loss, accuracy) = squat_detect_model.evaluate(xtest,ytest, batch_size=5, verbose=1)
print("[INFO] loss={:.4f}, accuracy: {:.4f}%".format(loss,accuracy * 100))

#Now the customized model(plastic_model) has been trained for 12 epochs using 12 training images
#Training accuracy=100%, Validation accuracy = 33.33%. This means this model with perfomr with 33.33% accuracy on new data

temp=squat_detect_model.predict(xtest)
print('accuracy score is =',accuracy_score(ytest.argmax(axis=1),temp.argmax(axis=1))*100,'%')
print(confusion_matrix(ytest.argmax(axis=1), temp.argmax(axis=1)))
print(classification_report(ytest.argmax(axis=1),temp.argmax(axis=1)))



train_loss=trained_data.history['loss']
val_loss=trained_data.history['val_loss']
train_acc=trained_data.history['acc']
val_acc=trained_data.history['val_acc']
xc=range(5)


plt.figure(1,figsize=(7,5))
plt.plot(xc,train_loss)
plt.plot(xc,val_loss)
plt.xlabel('num of Epochs')
plt.ylabel('loss')
plt.title('train_loss vs val_loss')
plt.grid(True)
plt.legend(['train','val'])
plt.style.use(['classic'])

plt.figure(2,figsize=(7,5))
plt.plot(xc,train_acc)
plt.plot(xc,val_acc)
plt.xlabel('num of Epochs')
plt.ylabel('accuracy')
plt.title('train_acc vs val_acc')
plt.grid(True)
plt.legend(['train','val'],loc=4)
plt.style.use(['classic'])

#Now saving the pretrained weights
# serialize model to JSON
model_json = trained_data.model.to_json()
with open("trained_data_squats.json", "w") as json_file:
    json_file.write(model_json)
# serialize weights to HDF5
trained_data.model.save_weights("trained_data_squats.h5")
print("Saved trained_data to disk")
    
