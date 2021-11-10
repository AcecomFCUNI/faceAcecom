# Escriba una secuencia de comandos de Python que capture imágenes de la cámara web
# Extrae todas las caras del marco de la imagen (usando haarcascades)
# Almacena la información de la cara en matrices numpy

# 1. Leer y mostrar el video, capturar imágenes
# 2. Detecta caras y muestra el cuadro delimitador (haarcascade)
# 3. Aplanar la imagen de la cara más grande (escala de grises) y guardar en una matriz numpy
# 4. Repita lo anterior para que varias personas generen datos de entrenamiento.

import cv2
import os
from os import getenv
from dotenv import load_dotenv  # pipenv install python-dotenv

load_dotenv()

HAAR_CASCADE = getenv("HAAR_CASCADE")
DATASET_DIR = getenv("DATASET_DIR")


# Init Camera
cap = cv2.VideoCapture(-1)  # 0

# Face Detection
face_cascade = cv2.CascadeClassifier(HAAR_CASCADE)

skip = 0
face_data = []
file_name = input("Ingrese el nombre de la persona: ")
if not os.path.exists(f"{DATASET_DIR}/{file_name}"):
    os.mkdir(f"{DATASET_DIR}/{file_name}")
while True:
    ret, frame = cap.read()
    if ret == False:
        continue

    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    cv2.imshow("Frame", frame)

    faces = face_cascade.detectMultiScale(frame, 1.3, 5)
    if len(faces) == 0:
        continue

    faces = sorted(faces, key=lambda f: f[2] * f[3])

    # Elija la última cara (porque es la cara más grande según el área (f[2]*f[3]))
    for face in faces[-1:]:
        x, y, w, h = face
        # cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 255), 2)

        # Extraer (Recortar la cara requerida): Región de interés
        offset = 10
        face_section = frame[y - offset : y + h + offset, x - offset : x + w + offset]
        face_section = cv2.resize(face_section, (160, 160))

        skip += 1
        if skip % 10 == 0:
            cv2.imwrite(
                f"{DATASET_DIR}/{file_name}/{file_name}_{skip}.jpg", face_section
            )
            print(f"Guardando {file_name}/{file_name}_{skip}", end="\r")

    key_pressed = cv2.waitKey(1) & 0xFF
    if key_pressed == ord("q"):
        break


print("Datos guardados en " + DATASET_DIR)

cap.release()
cv2.destroyAllWindows()
