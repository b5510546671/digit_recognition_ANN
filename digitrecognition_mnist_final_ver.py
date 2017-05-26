import numpy as np
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import Flatten
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import MaxPooling2D
from keras.preprocessing import image
from keras.utils import np_utils
from keras import backend as K
from scipy import ndimage, misc
import matplotlib.pyplot as plt
K.set_image_dim_ordering('th')


def baseline_model():
    # create model
    model = Sequential()
    model.add(Conv2D(32, (5,5), input_shape=(1,28,28), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2,2)))
    model.add(Dropout(0.2))
    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dense(num_classes, activation='softmax'))

    # compile model
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model

# load training and testing datasets
(X_train, y_train), (X_test, y_test) = mnist.load_data()

print ("1: {}".format(type(X_train)))
print ("11: {}".format(X_train.shape))
print ("11: {}".format(X_test.shape))
print ("11: {}".format(y_train.shape))
print ("11: {}".format(y_test.shape))

# reshape train/test dataset to be [samples][pixels][width][height]
X_train = X_train.reshape(X_train.shape[0],1,28,28).astype('float32')
X_test = X_test.reshape(X_test.shape[0],1,28,28).astype('float32')

print ("2: {}".format(type(X_train)))
print ("22: {}".format(X_train.shape))
print ("22: {}".format(X_test.shape))
print ("22: {}".format(y_train.shape))
print ("22: {}".format(y_test.shape))
# print ("22: {}".format(X_train))
# print ("22: {}".format(X_test))

# normalize inputs from 0-255 to 0-1
X_train = X_train / 255
X_test = X_test / 255

print ("3: {}".format(type(X_train)))
print ("33: {}".format(X_train.shape))
print ("33: {}".format(X_test.shape))
# print ("33: {}".format(X_train))
# print ("33: {}".format(X_test))

# one hot encode outputs
y_train = np_utils.to_categorical(y_train)
y_test = np_utils.to_categorical(y_test)
num_classes = y_test.shape[1]

# build the model
model = baseline_model()
# fit the model
model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=10, batch_size=200, verbose=2)
# final evaluation of the model
scores = model.evaluate(X_test, y_test, verbose=0)
print("Baseline Error: %.2f%%" % (100-scores[1]*100))

while True:

    photo_num = input("Enter image number: ")

    if str(photo_num) == "exit":
        break

    # try with our own digit image for prediction
    filepath = "C:\\Users\\knotsupavit\\Desktop\\ann\\neural2d\\images\\mnist\\train-data\\" + str(photo_num) + ".bmp"

    photo_predict = ndimage.imread(filepath, mode='L')

    print ("4: {}".format(type(photo_predict)))

    print ("5: {}".format(photo_predict.shape))

    # photo_predict = photo_predict.reshape(28,28)
    # photo_predict = photo_predict.reshape(photo_predict.shape[0],1,28,28).astype('float32')

    plt.imshow(photo_predict)
    plt.show()

    # f = np.load(filepath)
    print ("6: {}".format(type(photo_predict)))

    print ("7: {}".format(photo_predict.shape))

    photo_predict1 = np.array([[photo_predict]])
    photo_predict1 = photo_predict1 / 255
    print ("7.5: {}".format(photo_predict1.shape))

    # photo_predict1 = np_utils.to_categorical(photo_predict1)

    print ("8: {}".format(type(photo_predict1)))

    print ("9: {}".format(photo_predict1.shape))

    prediction = model.predict(photo_predict1)
    print(prediction)