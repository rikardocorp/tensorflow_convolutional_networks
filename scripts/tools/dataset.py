"""
AUTOR :
    Ricardo Coronado Pérez

DESCRIPCION:
    Esta clase a sido diseñada para manipular un conjunto de datos de imagenes,
    para ques esto funcione es necesario contar con un directorio de imagenes y 
    un archivo csv que contenga en la:

        - primera columna el nombre del archivo (imagen sin extension)
        - segunda columna la clase enumerada en un valor entero (ej. si contamos con 5 clases, 
          tendremos las clases enumeradas del 0 al 4)

    aunque el objetivo de esta clase es para cargar imagenes, puede modificarse el codigo para que pueda trabajar 
    con otro tipo de datos

FUNCIONES:
    generar bach : genera el primer conjunto de imagenes y sus etiquetas
    next batch : salta los indices de los punteros al siguiente conjunto de imagenes
    prev batch : salta los indices de los punteros al conjunto de datos anterior
    shuffler : re-ordena toda la data
"""

import os
import numpy as np
import pandas as pd
from tools.utils import show_image, load_image

class Dataset:
    """
        path_data: ruta del archivo csv que almacena el nombre de la imagen y su etiqueta
        path_dir_images: directorio donde se encuentran las imagenes
        minibatch: es el tamaño del conjunto de imagenes 
        cols: son los indices de las columnas del 'nombre de la imagen' y el 'label o clase'
        restrict: Retringe que el minibatch sea multiplo del total de imagenes
        random: Si es true automaticamente de reordenara al definir el objeto 'Dataset'
        xtype: Extension de las imagenes; si el nombre del archivo ya contiene la extension de la imagen podria asignarse '' vacio a este atributo,
        dim_image: Define la dimension de redimension para la imagen de entrada
        divisor_scale: Divisor que acotara los valores de la imagen entre [0, 1]
    """
    def __init__(
        self, 
        path_data='',
        path_dir_images='', 
        minibatch=25,
        cols=[],
        restrict=False,
        random=True,
        xtype=".jpg",
        dim_image=224,
        divisor_scale=255):

        assert os.path.exists(path_data), 'No existe el archivo con los datos de entrada ' + path_data

        self.path_data = path_data
        self.dir_images = path_dir_images
        self.minibatch = minibatch
        self.cols = cols
        self.type = xtype
        self.dim_image = dim_image
        self.divisor_scale = divisor_scale

        # leemos el archivo csv y guardamos las columnas 0 y 2 (nombre de imagen y etiqueta respectivamente)
        data = pd.read_csv(path_data, header=None)
        self.images = data[cols[0]]
        self.labels = data[cols[1]]
        self.total_images = len(data[cols[0]])
        # print(self.images)
        # print(self.labels)

        # inicializamos los punteros de la data
        self.start = 0
        self.end = minibatch

        if restrict is True:
            assert (self.total_images / self.minibatch).is_integer(), print(
                'El minibatch debe ser multiplo del total de datos de entrada.', self.total_images)

        # Considera solo batch completos
        self.total_batchs = int(self.total_images / self.minibatch)

        # Considera solo batch completos + 1 batch incompleto
        total_b = self.total_images / self.minibatch
        if (total_b - int(total_b)) > 0:
            self.total_batchs_complete = int(total_b) + 1
        else:
            self.total_batchs_complete = int(total_b)

        # Realizamos un reordenamiento por defecto
        if random is True:
            self.shuffler()
    
    # ##############################################################################
    # Reordena la lista de imagenes, simmula la aleatoridad en la eleccion de batchs
    def shuffler(self):
        cols = self.cols
        df = pd.read_csv(self.path_data, header=None)
        df = df.reindex(np.random.permutation(df.index))
        df = pd.DataFrame(df).reset_index(drop=True)
        self.images = df[cols[0]]
        self.labels = df[cols[1]]

    # ###############################################################################################  
    # Generamos el batch en la posicion actual donde se encuentras los punteros self.start y self.end
    def generate_batch(self):
        start = self.start
        end = self.end
        batch_list = []
        label_list = []
        # cargamos las imagenes y estas son tratadas para darles el tamaño requerido
        for i in range(start, end):
            # print(self.images[i], i)
            img = load_image(self.dir_images + str(self.images[i]) + self.type, scale=self.divisor_scale, dim_image=self.dim_image)[:, :, :3]
            batch_list.append(img.reshape((1, self.dim_image, self.dim_image, 3)))
            label_list.append(self.labels[i])

        # concatena cada elemento del batch_list dando como resultado una matriz con la forma (n, 224, 224, 3)
        # retorna el batch_list concatenado y el label_list con n items, una etiqueta para cada imagen
        return np.concatenate(batch_list, 0), label_list

    # ####################################################
    # Recorre la lista de imagenes de atras hacia adelante
    def next_batch(self):
        # es positivo cuando se llega al ultimo batch
        if (self.end / self.total_images) == 1 or ((self.total_images-self.end)/self.minibatch) < 1:
            # inicializa los indices y reordena la posicion de las imagenes
            self.start = 0
            self.end = self.minibatch
            self.shuffler()
        else:
            # hace que los indices apunte a las siguientes imagenes
            self.start = self.start + self.minibatch
            self.end = self.end + self.minibatch
    
    # ####################################################
    # Recorre la lista de imagenes de adelante hacia atras
    def prev_batch(self):
        # es positivo cuando el indice llega al primer batch
        if self.start == 0:
            # inicializa los indices para que apunten al bach final
            self.start = self.total_images - self.minibatch
            self.end = self.total_images
            self.shuffler()
        else:
            # actuliza los indices
            self.start = self.start - self.minibatch
            self.end = self.end - self.minibatch
    
    # ################################################################################################################    
    # Recorre de manera especial la lista para la fase de pruebas, cuando no existe la restriccion del minibatch
    def next_batch_test(self):
        dif = self.total_images - self.end
        dif_div = dif / self.minibatch
        if dif_div >= 1:
            self.start = self.start + self.minibatch
            self.end = self.end + self.minibatch
        elif dif_div == 0:
            self.start = 0
            self.end = self.minibatch
        elif dif_div < 1:
            self.start = self.start + self.minibatch
            self.end = self.total_images