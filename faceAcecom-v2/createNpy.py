import cv2
import numpy as np
import os
from os import getenv
from dotenv import load_dotenv  # pipenv install python-dotenv

load_dotenv()

HAAR_CASCADE = getenv("HAAR_CASCADE")
DATASET_DIR = getenv("DATASET_DIR")
NPY_DIR = getenv("NPY_DIR")


for idx, files in enumerate(os.walk(HAAR_CASCADE)):
    if idx > 0:
        face_data = []
        person_dir = files[0]
        for file in files[2]:
            image = cv2.cvtColor(
                cv2.imread(os.path.join(person_dir, file)), cv2.COLOR_BGR2RGB
            )
            image = cv2.resize(image, dsize=(160, 160))
            face_data.append(image)

        personName = person_dir.split("/")[2]
        face_data = np.asarray(face_data)
        face_data = face_data.reshape((face_data.shape[0], -1))
        np.save(f"{NPY_DIR}/{personName}.npy", face_data)
