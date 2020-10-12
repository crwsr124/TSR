import tensorflow as tf
from tensorflow import keras

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

        model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

        return model
