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

def cnn_model():
    model = Sequential()
    model.add(Conv2D(30, (5, 5), input_shape=(1, 28, 28), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(15, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.2))
    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dense(50, activation='relu'))
    model.add(Dense(num_classes, activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model

# load training and testing datasets
(image_data_train, label_train), (image_data_test, label_test) = mnist.load_data()

print ("1: {}".format(type(image_data_train)))
print ("11: {}".format(image_data_train.shape))
print ("11: {}".format(image_data_test.shape))
print ("11: {}".format(label_train.shape))
print ("11: {}".format(label_test.shape))

# reshape train/test dataset to be [samples][pixels][width][height]
image_data_train = image_data_train.reshape(image_data_train.shape[0],1,28,28).astype('float32')
image_data_test = image_data_test.reshape(image_data_test.shape[0],1,28,28).astype('float32')

print ("2: {}".format(type(image_data_train)))
print ("22: {}".format(image_data_train.shape))
print ("22: {}".format(image_data_test.shape))
print ("22: {}".format(label_train.shape))
print ("22: {}".format(label_test.shape))

# normalize inputs from 0-255 to 0-1
image_data_train = image_data_train / 255
image_data_test = image_data_test / 255

print ("3: {}".format(type(image_data_train)))
print ("33: {}".format(image_data_train.shape))
print ("33: {}".format(image_data_test.shape))

# one hot encode outputs
#transform vector of class integers into a binary matrix
label_train = np_utils.to_categorical(label_train)
label_test = np_utils.to_categorical(label_test)
num_classes = label_test.shape[1]


model = cnn_model()
# fit the model
model.fit(image_data_train, label_train, validation_data=(image_data_test, label_test), epochs=10, batch_size=200, verbose=2)
# final evaluation of the model
scores = model.evaluate(image_data_test, label_test, verbose=0)
print("Error: %.2f%%" % (100-scores[1]*100))

while True:

    input_file_number = input("Enter image number or 'exit' to exit: ")

    if str(input_file_number) == "exit":
        break

    elif input_file_number.isdigit():
        # try with our own digit image for prediction
        filepath = "C:\\Users\\knotsupavit\\Desktop\\ann\\neural2d\\images\\mnist\\validate-data\\" + str(input_file_number) + ".bmp"

        photo_predict = ndimage.imread(filepath, mode='L')

        print ("4: {}".format(type(photo_predict)))

        print ("5: {}".format(photo_predict.shape))

        # show image
        plt.imshow(photo_predict)
        plt.show()

        print ("6: {}".format(type(photo_predict)))

        print ("7: {}".format(photo_predict.shape))

        photo_predict_preprocessed = np.array([[photo_predict]])
        photo_predict_preprocessed = photo_predict_preprocessed / 255
        print ("7.5: {}".format(photo_predict_preprocessed.shape))


        print ("8: {}".format(type(photo_predict_preprocessed)))

        print ("9: {}".format(photo_predict_preprocessed.shape))

        prediction = model.predict(photo_predict_preprocessed).tolist()
        print(prediction[0])

        print(type(prediction[0]))

        count = 0
        max_num = -1000000000
        predict_num = 0
        for x in prediction[0]:
            if x > max_num:
                max_num = x
                predict_num = count
            count += 1

        print("Predict digit: {}".format(predict_num))

    else:
        print("Invalid input!")

