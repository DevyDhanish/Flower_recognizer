import tensorflow as tf
import numpy as np
from PIL import Image
import tensorflow_hub as hub
import matplotlib.pyplot as plt

image = Image.open("Test Images\\dandelion.jpg")
img_array = np.array(image)
IMG_RES = 224
class_names = ['dandelion','daisy','tulips','sunflowers','roses']

#model = tf.keras.models.load_model("tf_flower_87.h5")

with tf.keras.utils.custom_object_scope({"KerasLayer": hub.KerasLayer}):
    model = tf.keras.models.load_model('model\\tf_flower_87.h5')

def re_format(image):
  img = tf.image.resize(image, (IMG_RES, IMG_RES))/255.0
  return img

test_img = re_format(img_array)
test_img = np.array(test_img, dtype=float)

plt.imshow(test_img)

test_img = test_img.reshape((-1, 224, 224, 3))
output = model.predict(test_img)

title = class_names[np.argmax(output)]

print(title)

plt.title(f"I think these are {title}")
plt.show()
