from preproceso import preprocesses


def preproceso():
    directorio_entrada = './imagenes_entrenamiento'
    directorio_salida = './imagenes_procesadas'

    obj = preprocesses(directorio_entrada, directorio_salida)
    nde_imagenes_totales, nde_img_exitosamente_alineadas = obj.collect_data()

    print('Número total de imágenes: %d ' % nde_imagenes_totales)
    print('Número de imágenes alineadas y encuadradas correctamente: %d' % nde_img_exitosamente_alineadas)


if __name__ == '__main__':
    preproceso()
