from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import sys
from clasificador import training


def entrenamientoPrincipal():
    directorio_datos = './imagenes_procesadas'
    directorio_modelo = './modelo/modelo_preentrenado_caras.pb'
    nombre_clasificador = './clase/clasificador.pkl'
    print("Inicio del entrenamiento")
    obj = training(directorio_datos, directorio_modelo, nombre_clasificador)
    get_file = obj.main_train()
    print('Modelo clasificador guardado como "%s" ' % get_file)
    sys.exit("Proceso concluido satisfactoriamente ;)")


if __name__ == '__main__':
    entrenamientoPrincipal()