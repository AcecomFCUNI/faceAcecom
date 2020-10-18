import streamlit as st
import numpy as np
from PIL import Image, ImageEnhance
import os
import cv2
from identificar_cara_en_imagen import faceRecog

eyes=cv2.CascadeClassifier('./cascada/haarcascade_eye.xml')

def detect_eye(up_image):
	detect_img=np.array(up_image)
	new_img1=cv2.cvtColor(detect_img,1)
	faces=eyes.detectMultiScale(new_img1,1.3,5)
	for x,y,w,h in faces:
		cv2.rectangle(new_img1,(x,y),(x+w,y+h),(255,255,0),2)
	return new_img1,faces

def transform(up_image, type_t):

    if type_t == "Gris":
        new_img = np.array(up_image.convert('RGB'))
        gray = cv2.cvtColor(new_img, cv2.COLOR_BGR2GRAY)
        return gray
    if type_t == "Contraste":
        c_make = st.sidebar.slider("Contraste", 0.5, 3.5)
        enhacer = ImageEnhance.Contrast(up_image)
        img_out = enhacer.enhance(c_make)
        return img_out
    if type_t == "Brillo":
        b_make = st.sidebar.slider("Brillo", 0.5, 3.5)
        enhacer = ImageEnhance.Brightness(up_image)
        img_out = enhacer.enhance(b_make)
        return img_out 
    if type_t == "Desenfoque":
        br_make = st.sidebar.slider("Desenfoque", 0.5, 3.5)
        br_img = np.array(up_image.convert('RGB'))
        b_img = cv2.cvtColor(br_img, 1)
        blur = cv2.GaussianBlur(b_img, (11, 11), br_make)
        return blur
    if type_t == "Original":
        return up_image


page_bg_img = '''
<style>
    body {
        background-image: url('https://eskipaper.com/images/wallpapers-hd-37.jpg');
        background-size: cover;
    }
    h1 { 
        color: white; 
        text-shadow: 2px 3px 6px red; 
        } 
    .icons { 
        color: green; 
        width: 500px; 
        height: 200px; 
        border: 5px solid green; 
        } 
    i { 
        text-shadow:2px 4px 6px red; 
        } 
    .fa { 
        font-size:50px; 
        } 
    .fa-apple, .fa-car { 
        font-size:80px; 
    } 
</style>
'''
st.markdown(page_bg_img, unsafe_allow_html=True)


def main():

    st.title("faceAcecom v1")
    st.write("Construido con Tensorflow 1.4 y Streamlit")
    activites=["Predicción", "Acerca de"] #, "Identificación"
    choices=st.sidebar.selectbox("Seleccione opción",activites)


    if choices=="Predicción":
        st.subheader("Reconocimiento facial")
        img_file = st.file_uploader("Subir imagen", type=['png', 'jpg', 'jpeg'])
        if img_file is not None:
            up_image = Image.open(img_file)
            up_image.thumbnail((800, 800), Image.ANTIALIAS)
            st.image(up_image)
            enhace_type = st.sidebar.radio("Tipo", ["Original", "Gris", "Contraste", "Brillo", "Desenfoque"])
            transf_img = transform(up_image=up_image, type_t=enhace_type)
            st.image(transf_img)
        
        task = ["Cara", "Ojo"]
        feature_choice = st.sidebar.selectbox("Encontrar característica", task)
        if st.button("Ejecutar modelo"):
            if feature_choice == "Cara":
                result_img, result_faces = faceRecog(transf_img)
                st.image(result_img)
                st.success("{} caras encontradas".format(result_faces))
            if feature_choice=="Ojo":
                result_img, result_eyes = detect_eye(transf_img)
                st.image(result_img)
                st.success("{} ojos encontrados".format(len(result_eyes)))
        
    elif choices=="Acerca de":
        st.write("Esta aplicación fue creada el verano 2020 por el grupo ACECOM-IA")

if __name__ == '__main__':
    main()

