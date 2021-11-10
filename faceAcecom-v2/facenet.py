import cv2
import pickle
import numpy as np
from os import getenv
from dotenv import load_dotenv  # pipenv install python-dotenv

from architecture import InceptionResNetV2


load_dotenv()

FACENET_KERAS_WEIGHTS = getenv("FACENET_KERAS_WEIGHTS")
FACENET_PREDICTOR = getenv("FACENET_PREDICTOR")
LABEL_ENCODER = getenv("LABEL_ENCODER")


face_encoder = InceptionResNetV2()

# # Load the weights of the model
face_encoder.load_weights(FACENET_KERAS_WEIGHTS)
model_predictor = pickle.load(open(FACENET_PREDICTOR, "rb"))
model_label_encoder = pickle.load(open(LABEL_ENCODER, "rb"))


ids2name = {idx: name for idx, name in enumerate(model_label_encoder.classes_)}


def compute_embedding(model, face):
    face = face.astype("float32")
    mean, std = face.mean(), face.std()
    face = (face - mean) / std
    face = np.expand_dims(face, axis=0)

    embedding = model.predict(face)
    return embedding


def facenet(image):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, dsize=(160, 160))
    embedding = compute_embedding(face_encoder, image)
    prediction = model_predictor.predict(embedding)
    return ids2name[prediction[0]]


if __name__ == "__main__":
    img_test = cv2.imread("./dataset/Wiki/1.png")
    prediction = facenet(img_test)
    print(prediction)
