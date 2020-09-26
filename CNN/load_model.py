import tensorflow as tf

import numpy as np
from skimage import io
import matplotlib.pyplot as plt

from skimage import transform
from skimage import exposure
from skimage import io


# try to solve 
# "Could not create cudnn handle: CUDNN_STATUS_INTERNAL_ERROR"
from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession
config = ConfigProto()
config.gpu_options.allow_growth = True
session = InteractiveSession(config=config)


print("tensorflow version is:"+str(tf.__version__))

# load model
tsr_model = tf.keras.models.load_model('./model_result/tsr_model')
tsr_model.load_weights('./model_result/tsr_model_weights')

# test image
testimage = io.imread("7.jpg")
testimage = transform.resize(testimage,(32, 32, 3))
testimage = np.array(testimage)
print ("shape of image: " + str(testimage.shape))

# predict
predict_data = np.resize(testimage, (1, 32, 32, 3))
predict = tsr_model.predict (predict_data)
print ("the result is: " + str(predict))
print ("argmax is: " + str(np.argmax(predict)))

# show
plt.figure()
plt.imshow(testimage)
plt.show()
