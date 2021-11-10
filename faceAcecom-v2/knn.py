import numpy as np
import pickle
import os
import cv2
from os import getenv
from dotenv import load_dotenv  # pipenv install python-dotenv

load_dotenv()

NPY_DIR = getenv("NPY_DIR")
LABEL_ENCODER = getenv("LABEL_ENCODER")
model_label_encoder = pickle.load(open(LABEL_ENCODER, "rb"))
name2ids = {name: idx for idx, name in enumerate(model_label_encoder.classes_)}
ids2name = {idx: name for idx, name in enumerate(model_label_encoder.classes_)}


def loadData(npy_path=NPY_DIR):
    face_data = []
    labels = []

    # Preparación de datos
    for fx in os.listdir(npy_path):
        if fx.endswith(".npy"):
            data_item = np.load(npy_path + "/" + fx)
            face_data.append(data_item)

            # Crea etiquetas para la clase
            personName = fx.split(".")[-2]
            personId = name2ids[personName]
            target = personId * np.ones((data_item.shape[0],))
            labels.append(target)

    face_dataset = np.concatenate(face_data, axis=0)
    face_labels = np.concatenate(labels, axis=0).reshape((-1, 1))
    trainset = np.concatenate((face_dataset, face_labels), axis=1)
    return trainset


########## KNN ############
def distance(v1, v2):
    # Euclediana
    return np.sqrt(((v1 - v2) ** 2).sum())


def knn(train, test, k=5):
    dist = []

    for i in range(train.shape[0]):
        # Obtener el vector y la etiqueta
        ix = train[i, :-1]
        iy = train[i, -1]
        # Calcule la distancia desde el punto de prueba
        d = distance(test, ix)
        dist.append([d, iy])
    # Ordene según la distancia y obtenga la parte superior k
    dk = sorted(dist, key=lambda x: x[0])[:k]
    # Recuperar solo las etiquetas
    labels = np.array(dk)[:, -1]

    # Obtenga las frecuencias de cada etiqueta
    output = np.unique(labels, return_counts=True)
    # Encuentre la frecuencia máxima y la etiqueta correspondiente
    index = np.argmax(output[1])
    return output[0][index]


def knnPredictor(image, size=(160, 160)):
    trainset = loadData()
    face_section = cv2.resize(image, size)
    prediction = knn(trainset, face_section.flatten())
    return ids2name[prediction]


if __name__ == "__main__":
    img_test = cv2.imread("./dataset/Wiki/1.png")
    prediction = knnPredictor(img_test)
    print(prediction)
