""" $ pip install streamlit-webrtc opencv-python-headless matplotlib pydub
$ streamlit run app.py """

import logging
import threading
from pathlib import Path

import av
import cv2
import numpy as np
import pickle
from os import getenv
from dotenv import load_dotenv  # pipenv install python-dotenv
import streamlit as st

from streamlit_webrtc import (
    RTCConfiguration,
    VideoProcessorBase,
    WebRtcMode,
    webrtc_streamer,
)

from knn import knnPredictor
from SVM_HOG import svmHog
from facenet import facenet

load_dotenv()

HAAR_CASCADE = getenv("HAAR_CASCADE")
LABEL_ENCODER = getenv("LABEL_ENCODER")
HERE = Path(__file__).parent


logger = logging.getLogger(__name__)
face_cascade = cv2.CascadeClassifier(HAAR_CASCADE)
model_label_encoder = pickle.load(open(LABEL_ENCODER, "rb"))
name2ids = {name: idx for idx, name in enumerate(model_label_encoder.classes_)}
COLORS = np.random.uniform(0, 255, size=(len(model_label_encoder.classes_), 3))

RTC_CONFIGURATION = RTCConfiguration(
    {"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}
)


def main():
    st.header("FaceRecognition Models")

    knn_page = "Algoritmo K-NN (vecinos cercanos)"
    svm_hog_page = "Algoritmo HOG + SVM"
    facenet_page = "Red Neuronal FaceNet"

    app_mode = st.sidebar.selectbox(
        "Elige el modelo",
        [knn_page, svm_hog_page, facenet_page],
    )
    st.subheader(app_mode)

    if app_mode == knn_page:
        app_knn()
    elif app_mode == svm_hog_page:
        app_svm_hog()
    elif app_mode == facenet_page:
        app_facenet()

    st.sidebar.markdown(
        """
---
<a href="https://github.com/HiroForYou" target="_blank"><img src="https://github.com/HiroForYou/HiroForYou/raw/main/dev.svg" alt="Conoce mis repositorios" width="300" height="150" ></a>
    """,  # noqa: E501
        unsafe_allow_html=True,
    )

    logger.debug("=== Alive threads ===")
    for thread in threading.enumerate():
        if thread.is_alive():
            logger.debug(f"  {thread.name} ({thread.ident})")


def faceDetectBox(frame, model, offset=10):
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray_frame, 1.3, 5)
    if len(faces) == 0:
        return frame
    for face in faces:
        x, y, w, h = face
        face_section = frame[y - offset : y + h + offset, x - offset : x + w + offset]
        pred_name = model(face_section)
        cv2.rectangle(frame, (x, y), (x + w, y + h), COLORS[name2ids[pred_name]], 2)
        cv2.putText(
            frame,
            pred_name,
            (x, y - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            COLORS[name2ids[pred_name]],
            2,
            cv2.LINE_AA,
        )

    return frame


def app_knn():
    class knnVideoProcessor(VideoProcessorBase):
        def __init__(self) -> None:
            self._net = knnPredictor

        def recv(self, frame: av.VideoFrame) -> av.VideoFrame:
            image = frame.to_ndarray(format="bgr24")
            annotated_image = faceDetectBox(image, self._net)
            return av.VideoFrame.from_ndarray(annotated_image, format="bgr24")

    webrtc_ctx = webrtc_streamer(
        key="knn",
        mode=WebRtcMode.SENDRECV,
        rtc_configuration=RTC_CONFIGURATION,
        video_processor_factory=knnVideoProcessor,
        async_processing=True,
    )

    st.markdown(
        "El algoritmo k-nearest neighbors (KNN) es un algoritmo de aprendizaje automático simple y supervisado que se puede usar para resolver problemas de clasificación y regresión. Es fácil de implementar y entender, pero tiene un gran inconveniente de volverse significativamente lento a medida que crece el tamaño del dataset."
    )


def app_svm_hog():
    class hogVideoProcessor(VideoProcessorBase):
        def __init__(self) -> None:
            self._net = svmHog

        def recv(self, frame: av.VideoFrame) -> av.VideoFrame:
            image = frame.to_ndarray(format="bgr24")
            annotated_image = faceDetectBox(image, self._net)
            return av.VideoFrame.from_ndarray(annotated_image, format="bgr24")

    webrtc_ctx = webrtc_streamer(
        key="svm",
        mode=WebRtcMode.SENDRECV,
        rtc_configuration=RTC_CONFIGURATION,
        video_processor_factory=hogVideoProcessor,
        async_processing=True,
    )

    st.markdown(
        "Las características HOG son ampliamente utilizadas para la detección de objetos. HOG descompone una imagen en pequeñas celdas cuadradas, calcula un histograma de gradientes orientados en cada celda, normaliza el resultado utilizando un patrón de bloques y devuelve un descriptor para cada celda."
    )


def app_facenet():
    class facenetVideoProcessor(VideoProcessorBase):
        def __init__(self) -> None:
            self._net = facenet

        def recv(self, frame: av.VideoFrame) -> av.VideoFrame:
            image = frame.to_ndarray(format="bgr24")
            annotated_image = faceDetectBox(image, self._net)
            return av.VideoFrame.from_ndarray(annotated_image, format="bgr24")

    webrtc_ctx = webrtc_streamer(
        key="facenet",
        mode=WebRtcMode.SENDRECV,
        rtc_configuration=RTC_CONFIGURATION,
        video_processor_factory=facenetVideoProcessor,
        async_processing=True,
    )

    st.markdown(
        "FaceNet es un proyecto de reconocimiento facial desarrollado por tres investigadores de Google, Florian Schroff, Dmitry Kalenichenko y James Philbin en 2015. El objetivo principal de esta investigación es producir un embedding de la cara de una persona. Con arquitecturas de embedding como FaceNet, proyectas a las personas en un espacio de embeddings."
    )


if __name__ == "__main__":
    import os

    DEBUG = os.environ.get("DEBUG", "false").lower() not in ["false", "no", "0"]

    logging.basicConfig(
        format="[%(asctime)s] %(levelname)7s from %(name)s in %(pathname)s:%(lineno)d: "
        "%(message)s",
        force=True,
    )

    logger.setLevel(level=logging.DEBUG if DEBUG else logging.INFO)

    st_webrtc_logger = logging.getLogger("streamlit_webrtc")
    st_webrtc_logger.setLevel(logging.DEBUG)

    fsevents_logger = logging.getLogger("fsevents")
    fsevents_logger.setLevel(logging.WARNING)

    main()
