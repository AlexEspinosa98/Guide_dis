
from PyQt6.QtWidgets import QApplication, QMainWindow, QPushButton, QFileDialog
import os

import torch
import torchvision
from torchvision import transforms as torchtrans  
import torchvision.transforms as transforms

import cv2
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.patches as patches
from PIL import Image
import sqlite3

from datetime import datetime

"""/// Listado de funciones\\\
    self.ruta_variable => Variable que guarda la dir de folder
    self.ver_formato
"""

# Funcion para leer la dirección de carpeta
def ver_formato(ruta):
    lista= os. listdir(ruta)
    #print(lista)
    lis_jpg_tif=[]
    onl_jpg=[]
    con=0
    for nombres in lista:
            if nombres.split(".")[-1].upper() in ["jpg","JPG","TIF","tif"]:
                lis_jpg_tif.append(nombres)
            if nombres.split(".")[-1].upper() in ["jpg","JPG"]:
                onl_jpg.append(nombres)
        #variable nombres tiene solo los jpg y los tif
        #verificamos que por lo menos 1 tenga el formato
    #print(onl_jpg)
    for archivos in onl_jpg:
        if (os.path.exists(ruta+'/'+archivos[0:7]+'1.TIF')) and (os.path.exists(ruta+'/'+archivos[0:7]+'2.TIF')) and (os.path.exists(ruta+'/'+archivos[0:7]+'3.TIF')) and (os.path.exists(ruta+'/'+archivos[0:7]+'4.TIF')) and (os.path.exists(ruta+'/'+archivos[0:7]+'5.TIF')):
             con+=1
    return onl_jpg,con


#sección para realizar la predicción del modelo

def prediccion(ruta_imagen,model,lista_imagenes):
    # leemos las diferentes rutas de las imagenes en la función
    # Es decir, la ruta de cada imagen es ruta_imagen+lista_imagenes

    conn = sqlite3.connect('./library_new/test.db')
    cursor = conn.cursor()

    d = str(datetime.now())
        # Insertar datos en la tabla
    cursor.execute("INSERT INTO registro_carpeta (numero_de_imagenes,ruta_carpeta,fecha) VALUES (?, ?, ?)",
                    (str(len(lista_imagenes)),ruta_imagen, d))

        # Guardar los cambios y cerrar la conexión
    conn.commit()
    conn.close()


    # Verifivamos la id en la que esta 

    conn = sqlite3.connect('./library_new/test.db')
    cursor = conn.cursor()

    # Ejecutar la consulta SELECT *
    cursor.execute("SELECT * FROM registro_carpeta")

    # Obtener los resultados de la consulta
    resultados = cursor.fetchall()
    print(resultados)

    # Recorrer los resultados e imprimir los valores
    print(resultados[1][3])
    for indice,recorrido in enumerate(resultados):
        if (resultados[indice][3]==str(d)):
            id_datos=int(resultados[indice][0]) #ID de FK
    # Cerrar la conexión
    conn.close()
    #**********************************************# BASE DE DATOS TT
    for individual in lista_imagenes:
        ruta_total=ruta_imagen+"/"+individual
        print(ruta_total)
        image = Image.open(ruta_total)
        transform = transforms.Compose([
            transforms.Resize((1300, 1600)),
            transforms.ToTensor(),
            #transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Normalización típica de ImageNet
        ])
        input_image = transform(image).unsqueeze(0)  # Añade una dimensión adicional para el lote (batch)
        with torch.no_grad():  # Desactiva el cálculo de gradientes
            output = model(input_image)

        
        #llenamos base de datos
        conn = sqlite3.connect('./library_new/test.db')
        cursor = conn.cursor()

        d = str(datetime.now())
            # Insertar datos en la tabla
        cursor.execute("INSERT INTO tabla_imagenes (nombre_imagen,cantidad_detect,id_registro_carpeta) VALUES (?, ?, ?)",
                        (str(individual),int(len(output[0]['labels'])),int(id_datos)))

            # Guardar los cambios y cerrar la conexión
        conn.commit()
        conn.close()

#__ Id de segunda tabla (filtramos por ir de carpeta y nombre)

        conexion = sqlite3.connect('./library_new/test.db')
        cursor = conexion.cursor()

        # Definir los valores de nombre_imagen y id_registro_carpeta
        id_registro_carpeta = 1

        # Consultar el id de la fila con nombre_imagen y id_registro_carpeta especificados
        cursor.execute("SELECT id FROM tabla_imagenes WHERE nombre_imagen = ? AND id_registro_carpeta = ?", (str(individual), int(id_datos)))
        resultado = cursor.fetchone()

        if resultado is not None:
            id_fila = resultado[0] #id_fila tiene la id de guardado
        # Cerrar la conexión
        conexion.close()


        # Finalizacion llenado tabla 2 base de datos
        imagen_etiquetada=cv2.imread(ruta_total,1)
        imagen_etiquetada=cv2.cvtColor(imagen_etiquetada,cv2.COLOR_BGR2RGB)
        for detection in output:
            boxes = detection['boxes']
            labels = detection['labels']
            scores = detection['scores']
            
            
            # Recorre cada detección y dibuja el cuadro delimitador
            for box, label, score in zip(boxes, labels, scores):
                x, y, x2, y2 = box.tolist()
                
                #llenado de tabla 3
                conn = sqlite3.connect('./library_new/test.db')
                cursor = conn.cursor()

                d = str(datetime.now())
                    # Insertar datos en la tabla
                cursor.execute("INSERT INTO resultado_imagen (pixel_min,pixel_max,latitud,longitud,score,id_tabla_imagenes) VALUES (?, ?, ?)",
                                (str([x,y]),str([x2,y2]),lat,longi,str(score),id_fila))

                    # Guardar los cambios y cerrar la conexión
                conn.commit()
                conn.close()
                # FIN DE LLENADO DE TABLA 3
                #Nota: Tal vez lo de abajo ya no este

                cv2.rectangle(imagen_etiquetada, (int(x), int(y)), (int(x2), int(y2)), (0, 255, 0), 2)
                label_str = f'Class: {label.item()}, Score: {score.item():.2f}'
                cv2.putText(imagen_etiquetada, label_str, (int(x), int(y) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
                # Crea un objeto Rectangle para el cuadro delimitador
                """ rect = patches.Rectangle((x, y), w, h, linewidth=2, edgecolor='r', facecolor='none')
                
                # Añade el cuadro delimitador a la imagen
                ax.add_patch(rect)
                
                # Añade una etiqueta con la clase y la puntuación
                label_str = f'Clase: {label.item()}, score: {score.item():.2f}'
                ax.text(x, y, label_str, fontsize=8, color='r', verticalalignment='top') """
        
    return imagen_etiquetada


