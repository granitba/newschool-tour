# import the necessary packages
from PIL import Image
import numpy as np
import flask
import io
import os

# Disable TENSORFLOW WARNINGS
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import tensorflow as tf
from tensorflow import keras
import cv2
import imageio
import base64

# initialize our Flask application and the Keras model
app = flask.Flask(__name__)
model = None
from tensorflow.keras import layers

characters = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', 'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z']

# Map text to numeric labels
char_to_labels = {char:idx for idx, char in enumerate(characters)}

# Map numeric labels to text
labels_to_char = {val:key for key, val in char_to_labels.items()}


def decode_batch_predictions(pred):
    pred = pred[:, :-2]
    input_len = np.ones(pred.shape[0]) * pred.shape[1]

    # Use greedy search. For complex tasks, you can use beam search
    results = keras.backend.ctc_decode(pred,
                                       input_length=input_len,
                                       greedy=True)[0][0]

    # Iterate over the results and get back the text
    output_text = []
    for res in results.numpy():
        outstr = ''
        for c in res:
            if c < len(characters) and c >= 0:
                outstr += labels_to_char[c]
        output_text.append(outstr)

    # return final text results
    return output_text


def load_model():
    # load the pre-trained Keras model (here we are using a model
    # pre-trained on ImageNet and provided by Keras, but you can
    # substitute in your own networks just as easily)
    global model
    mod = tf.keras.models.load_model('captcha/saved_model/my_model')
    model = keras.models.Model(mod.get_layer(name='input_data').input, mod.get_layer(name='dense2').output)


def prepare_image(img):
    # if the image mode is not RGB, convert it
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = (img / 255.).astype(np.float32)
    img = cv2.medianBlur(img, 5)
    img = cv2.blur(img, (3, 3))
    kernel = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]])
    img = cv2.filter2D(src=img, ddepth=-2, kernel=kernel)
    img = np.array(img)
    img.shape
    img = img.T
    img.shape

    x = np.expand_dims(img, axis=-1)
    x = np.expand_dims(x, axis=0)
    return x

def predict_image(encoded_image: str):
    encoded_image.replace('', '+')
    image = imageio.v2.imread(io.BytesIO(base64.b64decode(encoded_image)))
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    # preprocess the image and prepare it for classification
    image = prepare_image(image)

    preds = model.predict(image)
    pred_texts = decode_batch_predictions(preds)

    return pred_texts


@app.route("/predict", methods=["POST"])
def predict():
    # initialize the data dictionary that will be returned from the
    # view
    data = {}
    # ensure an image was properly uploaded to our endpoint
    if flask.request.method == "POST":
        if flask.request.args.get("image"):
            print(flask.request.args.get("image"))
            st = flask.request.args.get("image")
            st = st.replace(' ', '+')
            print(st)
            image = imageio.v2.imread(io.BytesIO(base64.b64decode(st)))
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            # read the image in PIL format


            # preprocess the image and prepare it for classification
            image = prepare_image(image)

            # classify the input image and then initialize the list
            # of predictions to return to the client
            preds = model.predict(image)
            pred_texts = decode_batch_predictions(preds)

            # loop over the results and add them to the list of
            # returned predictions

            data['captchaResult'] = pred_texts
            # indicate that the request was a success

    # return the data dictionary as a JSON response
    return flask.jsonify(data)


if __name__ == "__main__":
    print(("* Loading Keras model and Flask starting server..."
           "please wait until server has fully started"))
    load_model()
    app.run(host='0.0.0.0', port=8080)  # Specify your desired port here
