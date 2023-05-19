import sys
from PyQt6.QtWidgets import QApplication, QMainWindow, QPushButton, QFileDialog
import os

"""/// Listado de funciones\\\
    self.ruta_variable => Variable que guarda la dir de folder

"""

# Funcion para leer la dirección de carpeta
def ver_formato(ruta):
    lista= os. listdir(ruta)
    print(lista)
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