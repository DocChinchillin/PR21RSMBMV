import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
import sys
import keras

from keras.utils.np_utils import to_categorical
from keras.layers import Convolution2D, MaxPooling2D, Dense, Dropout, Lambda, Flatten
from keras.models import Sequential
from keras.optimizers import Adam
from keras.preprocessing import image
from keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split
from koda import GUI_digit_rec

#loading test and train data


train = pd.read_csv('../podatki/train.csv')
test = pd.read_csv('../podatki/test.csv')

X_train = (train.iloc[:, 1:].values).astype('float32')  #vsi pixli
Y_train = train.iloc[:, 0].values.astype('int32') #target stevilke
X_test = test.values.astype('float32')


#data visual.

X_train = X_train.reshape(X_train.shape[0], 28, 28)


'''
for i in range(2,9):
    plt.subplot(330 + (i+1))
    plt.imshow(X_train[i])
    plt.title(Y_train[i])
plt.show()
'''

#ena dimenzija vec za barvo
X_train = X_train.reshape(X_train.shape[0], 28, 28, 1)
X_test = X_test.reshape(X_test.shape[0], 28, 28, 1)
print(X_train.shape)

#standarizacija ker TensorFlow deluje bolse ce so pixli standarizirani
mean_pixl = X_train.mean().astype(np.float32)
std_pikl = X_train.std().astype(np.float32)
def stand(x):
    return (x-mean_pixl)/std_pikl


#razredi -> one hot encoding
Y_train = to_categorical(Y_train)
num_of_classes = Y_train.shape[1]
#print(num_of_classes)

#ploting 20 element iz podatkov
plt.title(Y_train[20])
plt.plot(Y_train[20])
plt.xticks(range(0, 10))
#plt.show()



#Neural network CNN najboljse za slike

seed = 43
np.random.seed(seed)


#data split to train and test
X_train, X_val, Y_train, Y_val = train_test_split(X_train, Y_train, test_size=0.1, random_state=42)

def cnn_model():
    model = Sequential([
        Lambda(stand, input_shape=(28, 28, 1)),
        Convolution2D(32, (3, 3), activation='relu'),
        Convolution2D(32, (3, 3), activation='relu'),
        MaxPooling2D(),
        Convolution2D(64, (3, 3), activation='relu'),
        Convolution2D(64, (3, 3), activation='relu'),
        MaxPooling2D(),
        Flatten(),
        Dense(512, activation='relu'),
        Dense(10, activation='softmax')
    ])
    model.compile(Adam(), loss='categorical_crossentropy', metrics=['accuracy'])
    return model


# make cnn model and train it
'''
model = cnn_model()
model.optimizer.lr = 0.01
model.fit(X_train, Y_train, epochs=30, validation_data=(X_val, Y_val))
'''

# predict for kaggle

'''
pred = model.predict(X_test)
pred = np.argmax(pred, axis=1)
'''

# result for kaggle
'''
result = pd.DataFrame({'Id':list(range(1, len(pred)+1)), 'Label':pred}) 
result.to_csv('../podatki/res.csv', index=False)
'''


# save trained model
'''
tf.keras.models.save_model(model, '../podatki/model.h5py')
'''
import sys
from PyQt5 import QtCore, QtGui, QtWidgets, uic
from PyQt5.QtCore import Qt, QFile, QDataStream, QBuffer
from PyQt5.QtGui import QPen
from PyQt5.QtWidgets import (QWidget, QPushButton,
                             QHBoxLayout, QVBoxLayout, QApplication)
from PIL import Image
import PIL.ImageOps
import numpy as np

# load model
model = tf.keras.models.load_model('../podatki/model.h5py')


def predict_GUI_number(UI_model, image): #image is nump array
    image = image.reshape(-1, 28, 28, 1)
    pred = UI_model.predict([image])[0]
    return np.argmax(pred), max(pred)


# method to get number from gui
def get_image():
    img = window.label.grab().toImage().save('../podatki/num.jpg')
    img = Image.open('../podatki/num.jpg')
    img = img.resize((28, 28))
    img = img.convert('L')
    img_array = np.array(img)
    pred, acc = predict_GUI_number(model, img_array)

    window.pred_number.setText(f'Predicted number: {pred}, Accuracy {acc}')






# gui
app = QtWidgets.QApplication(sys.argv)
window = GUI_digit_rec.MainWindow()
window.show()



# number pred and acc
window.recoqnizebutton.clicked.connect(get_image)


app.exec()
