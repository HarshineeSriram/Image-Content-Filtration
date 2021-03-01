# Type the following to test:
# curl -X POST -F image=@Test_A001.png "http://127.0.0.1:5000/predict"
# where "Test_A001.png" is the complete filename

from keras.applications.mobilenet import preprocess_input
from keras.preprocessing.image import img_to_array
from keras import models


def load_model():
    global model
    model = models.load_model(r'path\to\model.h5')


def prepare_image(img):
    # Converts image to RGB,
    # resizes to 224X224 and
    # reshapes it for the MobileNet V1 Model
    if img.mode != "RGB":
        img = img.convert("RGB")
    img = img.resize((224, 224))
    img = img_to_array(img)
    img = img.reshape((1, img.shape[0], img.shape[1], 3))
    img = preprocess_input(img)
    return img
