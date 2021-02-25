# Type the following to test:
# curl -X POST -F image=@Test_A001.png "http://127.0.0.1:5000/predict"
# where "Test_A001.png" is the complete filename

from keras.applications.mobilenet import preprocess_input as p1
from keras.preprocessing.image import img_to_array
from keras import models
from PIL import Image
import numpy as np
import flask
import io

app = flask.Flask(__name__)
model = None


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
    img = p1(img)
    return img


@app.route("/predict", methods=["POST"])
def predict():
    data = {"success": False}
    if flask.request.method == "POST":
        if flask.request.files.get("image"):
            # Read the image and use model to predict
            # Store the predicted label and probability
            label = ""
            probability = 0

            image = flask.request.files["image"].read()
            image = Image.open(io.BytesIO(image))
            image = prepare_image(image)
            preds = model.predict(image)

            list_preds = list(preds)
            if(np.argmax(preds) == 0):
                label = "Unsafe"
                probability = list_preds[0][0]
            else:
                label = "Safe"
                probability = list_preds[0][1]

            data["predictions"] = []
            r = {"label": label, "probability": float(probability)}
            data["predictions"].append(r)
            data["success"] = True

    # return the data dictionary as a JSON response
    return flask.jsonify(data)


if __name__ == "__main__":
    print(("Loading the model and starting the server. . ."))
    load_model()
    app.run(debug=True, use_reloader=False)
