import pickle
import numpy as np
import cv2
from skimage.feature import hog
from os import getenv
from dotenv import load_dotenv  # pipenv install python-dotenv

load_dotenv()

SVM_HOG_PREDICTOR = getenv("SVM_HOG_PREDICTOR")
PCA_REDUCER = getenv("PCA_REDUCER")
LABEL_ENCODER = getenv("LABEL_ENCODER")


model_predictor = pickle.load(open(SVM_HOG_PREDICTOR, "rb"))
model_reductor = pickle.load(open(PCA_REDUCER, "rb"))
model_label_encoder = pickle.load(open(LABEL_ENCODER, "rb"))

ids2name = {idx: name for idx, name in enumerate(model_label_encoder.classes_)}


def svmHog(img, width=64, height=128):
    img = cv2.resize(img, (width, height))
    data_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    ppc = 8
    cb = 4
    # características, imagenes
    fd, _ = hog(
        data_gray,
        orientations=9,
        pixels_per_cell=(ppc, ppc),
        block_norm="L2",
        cells_per_block=(cb, cb),
        visualize=True,
    )
    hog_features = np.array(fd)
    hog_features = np.expand_dims(hog_features, axis=0)
    hog_features_pca = model_reductor.transform(hog_features)
    prediction = model_predictor.predict(hog_features_pca)
    return ids2name[prediction[0]]


if __name__ == "__main__":
    img_test = cv2.imread("./dataset/Wiki/1.png")
    prediction = svmHog(img_test)
    print(prediction)
