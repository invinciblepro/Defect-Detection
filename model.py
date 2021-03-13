import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import os
for dirname, _, filenames in os.walk('/casting_512x512'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

import tensorflow as tf
from tensorflow import keras#for back-end

from tensorflow.keras import Sequential#for sequentiality of layers

from tensorflow.keras.layers import Dense,Flatten,Conv2D,Dropout,MaxPool2D #for layers,operations and parameters

from tensorflow.keras.preprocessing.image import img_to_array  #to correctly fit the input into the sample data frame by using matrix

import cv2  #image lib

from PIL import Image #image lib

from tensorflow.keras import optimizers    # for graph optimation in training

import matplotlib.pyplot as plt

img_height=64

img_width=64

defective=os.listdir("casting_512x512/casting_512x512/def_front")

ok=os.listdir("casting_512x512/casting_512x512/ok_front")

label=[]

data=[]
i=0
#iterating through imgs directory
for img in defective :
    try :
        #reading each image
        img_read=plt.imread("casting_512x512/casting_512x512/def_front"+"/"+img)
        
        #resizing the images
        img_resize=cv2.resize(img_read,(img_height,img_width))
        
        #converting img to array
        img_array=img_to_array(img_resize)
        
        #taking img data
        data.append(img_array)
        
        if i<=10 :
            print(img_array)
        
        #taking ok or no values
        label.append(0)
    
    except :
        None
    i+=1
for img in ok :
    try :
        #reading each image
        img_read=plt.imread("casting_512x512/casting_512x512/ok_front"+"/"+img)
        
        #resizing the images
        img_resize=cv2.resize(img_read,(img_height,img_width))
        
        #converting img to array
        img_array=img_to_array(img_resize)
        
        #taking img data
        data.append(img_array)
        
        #taking ok or no values
        label.append(1)
    
    except :
        None
#converting data into numpy arrays
img_data=np.array(data)

label=np.array(label)

img_data=img_data/255
# Number of imgaes in first row print(img_data.shape[0])

#label array print(label)

#storing img data into x to get shuffle it
x=np.arange(img_data.shape[0])

#shuffling x
np.random.shuffle(x)

#mapping the original data with shuffled data
img_data=img_data[x]

label=label[x]

print(img_data.shape)
print(label)

from sklearn.model_selection import train_test_split

#80 are for training and 20 are for testing in every 101 samples
x_train,x_test,y_train,y_test=train_test_split(img_data,label,test_size=0.2,random_state=101)

print(x_train.shape,x_test.shape,y_train.shape,y_test.shape)

#INPUT LAYER

#creating model
model=Sequential()

#CNN Usage

#for an input layer
model.add(Conv2D(16,(3,3),input_shape=(img_height,img_width,3),activation='relu'))

#normalization
model.add(MaxPool2D(2,2))

model.add(Dropout(0.2))

#Brings data from 3x3 to one dimension
model.add(Flatten())


#HIDDEN LAYERS 1

model.add(Dense(64,activation='relu'))

#model.add(MaxPool2D(2,2))

model.add(Dropout(0.2))

#HIDDEN LAYER 2

model.add(Dense(64,activation='relu'))

#model.add(MaxPool2D(2,2))

model.add(Dropout(0.2))

#HIDDEN LAYER 3
model.add(Dense(64,activation='relu'))

#model.add(MaxPool2D(2,2))

model.add(Dropout(0.5))

#OUTPUT LAYER
#sigmoid is a logistic function----->0 or 1 as output
model.add(Dense(1,activation='sigmoid'))

model.summary()


#compiling model
model.compile(optimizer='adam',loss=tf.keras.losses.BinaryCrossentropy(),metrics=['accuracy'])

#model.evaluate(x_test,y_test,batch_size=128,verbose=1)
res_model=model.fit(x_train, y_train, batch_size=50, epochs=40, verbose=1)

#testing for the data (x_test,y_test)

predictions=model.evaluate(x_test,y_test)

#tf.keras.callbacks.Histroy()

#frame
plt.figure(figsize=(7,5))

acc=res_model.history['accuracy']

loss=res_model.history['loss']

plt.plot(acc)

plt.xlabel('number of epochs')

plt.ylabel('accuracy')

plt.title('Training accuracy')

plt.show()


plt.plot(loss)

plt.xlabel('number of epochs')

plt.ylabel('loss')

plt.title('Training loss')

plt.show()

# serialize to JSON
json_file = model.to_json()
with open("model.json", "w") as file:
   file.write(json_file)
# serialize weights to HDF5
model.save_weights("model.h5")