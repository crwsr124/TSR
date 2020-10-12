import tensorflow as tf
import numpy as np
import pathlib
import PIL
import matplotlib.pyplot as plt


print("----------------------------begin of code\ntensorflow version:" + str(tf.__version__))

imgs_nolabel = pathlib.Path("./imgs_nolabel")

img_paths = list(imgs_nolabel.glob("*"))

print (str(img_paths[0]))
print (len(img_paths))

img = PIL.Image.open(str(img_paths[0]))
img.show()

# plt.figure("dog")
# plt.imshow(img)
# plt.show()
