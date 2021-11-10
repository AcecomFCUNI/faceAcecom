<p align="center">
    <br>
    <a href="https://www.facebook.com/acecom.uni">
    <img src="https://i.imgur.com/SPrRIfs.png"/>
    </a>
    <br>
</p>

<h2 align="center">
<p>faceAcecom V1 😄</p>
</h2>
Construido con Tensorflow 1.4 y Streamlit el verano 2020. 


<p align="center">
  <img src="steps.gif" />
</p>

Este proyecto contó con la participación de:
* Brando Palacios, Github: [BrandoPM18](https://github.com/BrandPM18)
* Cristhian Wiki, Github: [HiroForYou](https://github.com/HiroForYou)
* Nelson S. Maldonado
* Diego Bayes, Github: [FixerDiegoB](https://github.com/FixerDiegoB)
* Julissa García, Github: [ajulissa](https://github.com/ajulissa)
* Angélica Sánchez, Github: [Angelica-17](https://github.com/Angelica-17)

## 📑 Instalación de los paquetes
* Para poder ejecutar este proyecto le recomendamos encarecidamente use ANACONDA o *python venv* para poder manejar sus entornos de trabajo y no tener problemas de incompatibilidad.
* Una vez que tiene su entorno creado y **activado**, ejecute el siguiente comando en terminal:
```bash
pip install -r requirements.txt
```
Esto puede tomar un poco de tiempo, asi que tenga paciencia!
* Eso es todo, estas listo para pasar a la siguiente sección.

**NOTA**: Usted también puede implementar contenedores Docker para agilizar más el proceso de despliegue del modelo.

## 💻 Instrucciones
Si usted desea implementar el modelo entrenado que hemos creado, siga los siguientes pasos:
* Descargue los 3 archivos *npy* en el siguiente [enlace](https://drive.google.com/drive/folders/1JHcal6ohbjRaebm0ZMIdwIiV6FDzX3kb?usp=sharing) y coloquelos en la carpeta **npy**.
* Descargue el archivo [modelo_preentrenado_caras.pb](https://drive.google.com/file/d/1WBgqnlunACpfIOzsG3xStzqIGY1t_lAP/view?usp=sharing) y colóquelo en la carpeta **modelo**.
* Desde **terminal** ejecute:
```bash
streamlit run RUN.py
```
A continuación solo seleccione la imagen de su preferencia en el navegador.

Si usted desea entrenar el modelo en su propio conjunto de datos:
* Cree en la carpeta **imagenes_entrenamiento** las subcarpetas con los nombres de los individuos. Para un buen rendimiento se recomienda 30 fotos por persona. **DETALLE**: TODAS LAS IMÁGENES EN 3 CANALES DE COLOR, SINO LANZARÁ ERROR 'imagen no encontrada'.
* Para poder preprocesar los datos y solo entrenar con imágenes de rostros encuadrados, ejecute desde **terminal**: 
```bash
python preproceso.py
```
* Descargue los 3 archivos *npy* en el siguiente [enlace](https://drive.google.com/drive/folders/1JHcal6ohbjRaebm0ZMIdwIiV6FDzX3kb?usp=sharing) y coloquelos en la carpeta **npy**.
* Descargue el archivo [modelo_preentrenado_caras.pb](https://drive.google.com/file/d/1WBgqnlunACpfIOzsG3xStzqIGY1t_lAP/view?usp=sharing) y colóquelo en la carpeta **modelo**.
* Finalmente para entrenar el modelo en su dataset (no requiere mucho tiempo) ejecute desde **terminal**:
```bash
python entrenamiento_principal.py
```
* Con su modelo ya entrenado, puede realizar inferencias con los archivos `python identificar_cara_en_imagen.py`, `python identificar_cara_video_vivo.py`, `python identificar_cara_video.py` de acuerdo a las necesidades que tenga. También puede ejecutar `streamlit run RUN.py` para lanzar la aplicación y probarla en su nuevo modelo.

**NOTA:** El tiempo de inferencia dependerá de si usted posee una GPU con CUDA instalado y TF-GPU v 1.*. Generalmente demora la inferencia en CPU.

## 📘 Materiales adicionales
Puede encontrar todo el material adicional que consultamos, creamos y corregimos en este [enlace](https://drive.google.com/drive/folders/1Ib4MYnbTmBygGVlv0b-ShhwqlJKIc3VG?usp=sharing). Todo el material es gratuito para que usted mismo lo pueda replicar en sus propios proyectos.
Cualquier consulta puede escribir a h3artcalcif3r@gmail.com, le responderemos lo más antes posible. 