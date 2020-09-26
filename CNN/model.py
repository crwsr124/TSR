import tensorflow as tf
from tensorflow import keras

import numpy as np
from skimage import io
import matplotlib.pyplot as plt

from skimage import transform
from skimage import exposure
from skimage import io

import sys
import random

# try to solve 
# "Could not create cudnn handle: CUDNN_STATUS_INTERNAL_ERROR"
from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession
config = ConfigProto()
config.gpu_options.allow_growth = True
session = InteractiveSession(config=config)


print("tensorflow version is:"+str(tf.__version__))


def getTrainData(csv_path):
    data = []
    labels = []
    rows = open(csv_path).read().strip().split("\n")[1:]
    random.shuffle(rows)
    print("train_meta_data:")
    print(rows[0])
    print(rows[1])
    print(rows[2])
    print ("......")

    i = 0
    for row_i in rows:

        if i > -1 :#and i < 1000:
            (label, imagePath) = row_i.strip().split(",")[-2:]
            image = io.imread("./82373_191501_bundle_archive/"+imagePath)
    
            # plt.figure(0)
            # plt.imshow(image)
            # plt.show()

            # resize image and 
            image = transform.resize(image,(32, 32, 3),preserve_range=True)/255.0
            
            # image = image*1.5
            # plt.figure(1)
            # plt.imshow(image)
            # plt.show()

            # image = image/1.5
            # image = exposure.equalize_adapthist(image, clip_limit=0.1)
            # plt.figure(2)
            # plt.imshow(image)
            # plt.show()

            data.append(image)
            labels.append(int(label))
        
        i = i+1

    data = np.array(data)
    labels = np.array(labels)

    return data, labels

def getTestData(csv_path):
    data = []
    labels = []
    rows = open(csv_path).read().strip().split("\n")[1:]
    print("test_meta_data:")
    print(rows[0])
    print(rows[1])
    print(rows[2])
    print ("......")

    for row_i in rows:
            (label, imagePath) = row_i.strip().split(",")[-2:]
            image = io.imread("./82373_191501_bundle_archive/"+imagePath)

            image = transform.resize(image,(32, 32, 3))

            data.append(image)
            labels.append(int(label))

    data = np.array(data)
    labels = np.array(labels)

    return data, labels
	

    



class TrafficSignRecognition:

    param1 = 0
    param2 = 0
    param3 = 0

    def __init__(self):
        self.param1 = 0
        self.param2 = 0
        self.param3 = 0
        print("######################################\nmodel init with param:")


    def build(self, input_width, input_height, input_channel, block_num, class_num):
        print("building")
        
        # model = keras.Sequential([
        #         keras.layers.Flatten(input_shape=(input_width, input_height, input_channel)),
        #         keras.layers.Dense(128, activation='relu'),
        #         keras.layers.Dense(class_num, activation='relu')
        # ])

        # model = keras.Sequential()
        # model.add( keras.layers.Conv2D(43, (3,3), padding='same', activation='relu', input_shape=(32, 32, 3)) )
        # model.add( keras.layers.Conv2D(43, (3,3), padding='same', activation='relu') )
        # l1 = keras.layers.Conv2D(43, (3,3), padding='valid', activation='relu')
        # model.add( l1 )
        # model.add( keras.layers.Conv2D(43, (3,3), padding='same', activation='relu') )
        # l2 = keras.layers.Conv2D(43, (3,3), padding='same', activation='relu')
        # model.add( tf.keras.layers.add([l1, l2]) )
        # model.add( keras.layers.Conv2D(43, (3,3), padding='valid', activation='relu') )
        # model.add( keras.layers.Conv2D(43, (3,3), padding='same', activation='relu') )
        # model.add( keras.layers.Conv2D(43, (3,3), padding='same', activation='relu') )
        # model.add( keras.layers.Conv2D(43, (3,3), padding='valid', activation='relu') )
        # model.add( keras.layers.Conv2D(43, (3,3), padding='same', activation='relu') )
        # model.add( keras.layers.Conv2D(43, (3,3), padding='same', activation='relu') )
        # model.add( keras.layers.Conv2D(43, (3,3), padding='same', activation='relu') )
        # model.add( keras.layers.GlobalAveragePooling2D() )
        # model.add( keras.layers.Flatten() )
        # model.add( keras.layers.Dense(128, activation='relu') )
        # model.add( keras.layers.Dense(class_num, activation='relu') )

        inputs = tf.keras.Input(shape=(32, 32, 3), name="input")
        c1 = keras.layers.Conv2D(86, (3,3), padding='same', activation='relu')(inputs)
        c1 = tf.keras.layers.BatchNormalization()(c1)
        c2 = c1 + keras.layers.Conv2D(86, (3,3), padding='same', activation='relu')(c1)
        c2 = tf.keras.layers.BatchNormalization()(c2)
        c3 = keras.layers.Conv2D(86, (3,3), padding='valid', activation='relu')(c2)
        c3 = tf.keras.layers.BatchNormalization()(c3)
        c4 = keras.layers.Conv2D(86, (3,3), padding='same', activation='relu')(c3)
        c4 = tf.keras.layers.BatchNormalization()(c4)
        c5 = c3 + keras.layers.Conv2D(86, (3,3), padding='same', activation='relu')(c4)
        c5 = tf.keras.layers.BatchNormalization()(c5)
        c6 = keras.layers.Conv2D(86, (3,3), padding='valid', activation='relu')(c5)
        c6 = tf.keras.layers.BatchNormalization()(c6)
        c7 = keras.layers.Conv2D(86, (3,3), padding='same', activation='relu')(c6)
        c7 = tf.keras.layers.BatchNormalization()(c7)
        c8 = keras.layers.Conv2D(86, (3,3), padding='same', activation='relu')(c7)
        c8 = tf.keras.layers.BatchNormalization()(c8)
        c9 = c6 + keras.layers.Conv2D(86, (3,3), padding='same', activation='relu')(c8)
        c9 = tf.keras.layers.BatchNormalization()(c9)
        c10 = keras.layers.GlobalAveragePooling2D()(c9)
        c11 = keras.layers.Flatten()(c10)
        #c12 = keras.layers.Dense(128, activation='relu')(c11)
        c13 = keras.layers.Dense(class_num, activation='relu')(c11)
        model = keras.Model(inputs=inputs, outputs=c13, name='TSR')

        # model = keras.Sequential()
        # model.add( keras.layers.Conv2D(76, (3,3), padding='same', activation='relu', input_shape=(32, 32, 3)) )
        # model.add( keras.layers.Conv2D(76, (3,3), padding='same', activation='relu') )
        # model.add( keras.layers.Conv2D(72, (3,3), padding='valid', activation='relu') )
        # model.add( keras.layers.Conv2D(72, (3,3), padding='same', activation='relu') )
        # model.add( keras.layers.Conv2D(72, (3,3), padding='same', activation='relu') )
        # model.add( keras.layers.Conv2D(68, (3,3), padding='valid', activation='relu') )
        # model.add( keras.layers.Conv2D(68, (3,3), padding='same', activation='relu') )
        # model.add( keras.layers.Conv2D(68, (3,3), padding='same', activation='relu') )
        # model.add( keras.layers.Conv2D(64, (3,3), padding='valid', activation='relu') )
        # model.add( keras.layers.Conv2D(64, (3,3), padding='same', activation='relu') )
        # model.add( keras.layers.Conv2D(64, (3,3), padding='same', activation='relu') )
        # model.add( keras.layers.Conv2D(64, (3,3), padding='same', activation='relu') )
        # model.add( keras.layers.Conv2D(64, (26,26), padding='valid', activation='relu') )
        # model.add( keras.layers.Flatten() )
        # model.add( keras.layers.Dense(128, activation='relu') )
        # model.add( keras.layers.Dense(class_num, activation='relu') )



        model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

        return model


train_data, labels = getTrainData("./82373_191501_bundle_archive/Train.csv")
test_data, test_labels = getTrainData("./82373_191501_bundle_archive/Test.csv")

print ("train_shape: " + str(train_data.shape))
print ("labels_shape: " + str(labels.shape))

tsr = TrafficSignRecognition()
model = tsr.build(32, 32, 3, 3, 43)
model.summary()

#sys.exit()

model.fit(train_data, labels, batch_size=6, epochs=30)
test_loss, test_acc = model.evaluate(test_data,  test_labels, verbose=2)

print('\nTest accuracy:', test_acc)

model.save('./model_result/tsr_model') 
model.save_weights('./model_result/tsr_model_weights')


# testimage = io.imread("1.jpg")
# testimage = transform.resize(testimage,(32, 32, 3))
# testimage = np.array(testimage)
# print ("shape of image: " + str(testimage.shape))

# predict_data = np.resize(testimage, (1, 32, 32, 3))
# predict = model.predict (predict_data)
# print ("the result is: " + str(predict))
# print ("argmax is: " + str(np.argmax(predict)))



# plt.figure()
# plt.imshow(testimage)
# plt.show()

