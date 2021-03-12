import io
import flask
import numpy as np
from constants_api import (
    debug, use_reloader
)
from prepare_image_api import (
    load_model, prepare_image
)
from PIL import Image

app = flask.Flask(__name__)
model = load_model()


@app.route("/predict", methods=["POST"])
def predict():
    data = {}
    if flask.request.files.get("image"):
        # Read the image and use model to predict
        # Store the predicted label and probability

        image = flask.request.files["image"].read()
        image = Image.open(io.BytesIO(image))
        image = prepare_image(image)
        preds = model.predict(image)

        list_preds = list(preds)

        label = "Unsafe" if np.argmax(preds) == 0 else "Safe"
        probability = list_preds[0][0] if np.argmax(preds) == 0 \
            else list_preds[0][1]

        r = {"label": label, "probability": float(probability)}
        data["predictions"].append(r)

    # return the data dictionary as a JSON response and the status code
    return data, 200


if __name__ == "__main__":
    print(("Loading the model and starting the server. . ."))
    app.run(debug=debug, use_reloader=use_reloader)
