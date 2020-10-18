from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


import tensorflow as tf
from scipy import misc
import cv2
import numpy as np
import facenet
import detect_face
# import pronunciar
import os
import time
import pickle
import pyaudio
import wave
import sys

def videoVivoRec():
    CHUNK = 1024

    dir_voces = './voces/'
    nombres_hablados = []
    # angelica = wave.open(f'{dir_voces}/angelica.wav', 'rb')
    # angelica = wave.open(f'{dir_voces}/angelica.wav', 'rb')
    # voces = []

    modeldir = './modelo/modelo_preentrenado_caras.pb'
    classifier_filename = './clase/clasificador.pkl'
    npy = './npy'
    train_img = "./imagenes_entrenamiento"


    def speech(wf=''):
        p = pyaudio.PyAudio()

        stream = p.open(format=p.get_format_from_width(wf.getsampwidth()),
                        channels=wf.getnchannels(),
                        rate=wf.getframerate(),
                        output=True)

        data = wf.readframes(CHUNK)

        while len(data) > 0:
            stream.write(data)
            data = wf.readframes(CHUNK)

        stream.stop_stream()
        stream.close()

        p.terminate()


    with tf.Graph().as_default():
        gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.4)
        sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options, log_device_placement=False))
        with sess.as_default():
            pnet, rnet, onet = detect_face.create_mtcnn(sess, npy)

            minsize = 20  # tamaño mínimo de la cara
            threshold = [0.6, 0.7, 0.7]  # umbral de tres pasos
            factor = 0.709  # factor de escala
            margin = 44
            frame_interval = 3
            batch_size = 1000
            image_size = 182
            input_image_size = 160

            HumanNames = os.listdir(train_img)
            HumanNames.sort()

            print('Cargando modelo')
            facenet.load_model(modeldir)
            images_placeholder = tf.get_default_graph().get_tensor_by_name("input:0")
            embeddings = tf.get_default_graph().get_tensor_by_name("embeddings:0")
            phase_train_placeholder = tf.get_default_graph().get_tensor_by_name("phase_train:0")
            embedding_size = embeddings.get_shape()[1]

            classifier_filename_exp = os.path.expanduser(classifier_filename)
            with open(classifier_filename_exp, 'rb') as infile:
                (model, class_names) = pickle.load(infile)

            video_capture = cv2.VideoCapture(0)  # cambiado a canal 1 (celular)
            c = 0

            print('Comenzando Reconocimiento :D!')
            prevTime = 0
            while True:
                ret, frame = video_capture.read()

                frame = cv2.resize(frame, (0, 0), fx=1, fy=1)  # redimensionar frame (opcional)

                curTime = time.time() + 1  # calculando fps
                timeF = frame_interval

                if (c % timeF == 0):
                    find_results = []

                    if frame.ndim == 2:
                        frame = facenet.to_rgb(frame)
                    frame = frame[:, :, 0:3]
                    bounding_boxes, _ = detect_face.detect_face(frame, minsize, pnet, rnet, onet, threshold, factor)
                    nrof_faces = bounding_boxes.shape[0]
                    print('Número de caras detectadas: %d' % nrof_faces)

                    if nrof_faces > 0:
                        det = bounding_boxes[:, 0:4]
                        img_size = np.asarray(frame.shape)[0:2]

                        cropped = []
                        scaled = []
                        scaled_reshape = []
                        bb = np.zeros((nrof_faces, 4), dtype=np.int32)

                        for i in range(nrof_faces):
                            emb_array = np.zeros((1, embedding_size))

                            bb[i][0] = det[i][0]
                            bb[i][1] = det[i][1]
                            bb[i][2] = det[i][2]
                            bb[i][3] = det[i][3]

                            # Excepción interna
                            if bb[i][0] <= 0 or bb[i][1] <= 0 or bb[i][2] >= len(frame[0]) or bb[i][3] >= len(frame):
                                print('La cara esta muy cerca!')
                                continue

                            cropped.append(frame[bb[i][1]:bb[i][3], bb[i][0]:bb[i][2], :])
                            cropped[i] = facenet.flip(cropped[i], False)
                            scaled.append(misc.imresize(cropped[i], (image_size, image_size), interp='bilinear'))
                            scaled[i] = cv2.resize(scaled[i], (input_image_size, input_image_size),
                                                   interpolation=cv2.INTER_CUBIC)
                            scaled[i] = facenet.prewhiten(scaled[i])
                            scaled_reshape.append(scaled[i].reshape(-1, input_image_size, input_image_size, 3))
                            feed_dict = {images_placeholder: scaled_reshape[i], phase_train_placeholder: False}
                            emb_array[0, :] = sess.run(embeddings, feed_dict=feed_dict)
                            predictions = model.predict_proba(emb_array)
                            print(predictions)
                            best_class_indices = np.argmax(predictions, axis=1)
                            best_class_probabilities = predictions[np.arange(len(best_class_indices)), best_class_indices]
                            # print("predicciones")
                            print(best_class_indices, ' con una precisión de ', best_class_probabilities)

                            # print(best_class_probabilities)
                            if best_class_probabilities > 0.53:
                                cv2.rectangle(frame, (bb[i][0], bb[i][1]), (bb[i][2], bb[i][3]), (66, 153, 236),
                                              1)  # encajando cara

                                # plotear resulto idx debajo de la caja
                                text_x = bb[i][0]
                                text_y = bb[i][3] + 20
                                prob_x = bb[i][0] + 15
                                prob_y = bb[i][1] - 10
                                print('Índices de resultados: ', best_class_indices[0])
                                print(HumanNames)
                                for H_i in HumanNames:
                                    if HumanNames[best_class_indices[0]] == H_i:
                                        result_names = HumanNames[best_class_indices[0]]
                                        dec = np.round(best_class_probabilities, 4)
                                        cv2.putText(frame, str(dec), (prob_x, prob_y), cv2.FONT_ITALIC,
                                                    0.5, (30, 103, 202), thickness=1, lineType=1)
                                        cv2.putText(frame, result_names, (text_x, text_y), cv2.FONT_HERSHEY_DUPLEX,
                                                    0.7, (8, 6, 98), thickness=1, lineType=0)
                                        wf = wave.open(f'{dir_voces}/{result_names}.wav', 'rb')
                                        # print('longitud: ' , len(nombres_hablados))
                                        if result_names not in nombres_hablados:
                                            nombres_hablados.append(result_names)
                                            speech(wf)
                                            print('hola')
                    else:
                        print('Fallo de alineación')
                # c+=1
                marco_display = cv2.resize(frame, (1200, 650), interpolation=cv2.INTER_CUBIC)
                cv2.imshow('Detectando rostros en vivo..', marco_display)

                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

            video_capture.release()
            cv2.destroyAllWindows()


if __name__ == '__main__':
    videoVivoRec()