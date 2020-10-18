<p align="center">
    <br>
    <a href="https://www.facebook.com/acecom.uni">
    <img src="https://i.imgur.com/SPrRIfs.png"/>
    </a>
    <br>
</p>

<h2 align="center">
<p>Sistema de Identificación Facial (faceAcecom) 👀💻</p>
</h2>

Proyecto interno desarrollo en ACECOM-IA para el monitoreo y verificación de las identidades faciales de los miembros.

## Actualización
- 18/10/20: faceAcecomV1 ahora se ejecuta desde un servidor local con **streamlit**. También se ha añadido soporte para filtros de imagen, y ahora también es capaz de reconocer otros componentes del rostro: ojos. 

## 📖 Contenido
El siguiente árbol muestra la estructura de las versiones de la aplicación:
```
|- master/
|   |- faceAcecomV1/
|       |- cascada/
|           |- haarcascade_eye.xml 
|       |- clase/
|       |- imagenes_entrenamiento/
|       |- imagenes_procesadas/
|       |- modelo/
|           |- modelo_preentrenado_caras.pb
|       |- npy/
|           |- det1.npy
|           |- det2.npy
|           |- det3.npy
|       |- voces/
|       |- clasificador.py   
|       |- detect_face.py  
|       |- entrenamiento_principal.py  
|       |- identificar_cara_en_imagen.py  
|       |- identificar_cara_video_vivo.py  
|       |- identificar_cara_video.py  
|       |- preproceso_datos.py  
|       |- preproceso.py  
|       |- RUN.py  
|       |- README.md
|   |- faceAcecomV2/
|       |- __init__.py   
|       |- README.md
```
## Maintainers
* Cristhian Wiki, github: [HiroForYou](https://github.com/HiroForYou)

## Agradecimientos
* Versión 1:
Muchas gracias a los miembros de ACECOM-IA por el compromiso presentado en el proyecto durante casi un mes. 
Las exposiciones y *DEBUGEOS* no hubiesen sido posible sin el apoyo en equipo.
* Versión 2:
*próximamente*