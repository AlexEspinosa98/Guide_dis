
from PyQt6.QtWidgets import QMainWindow, QApplication,QLineEdit,QMessageBox,QTableWidget,QFileDialog,QTableWidgetItem

from PyQt6.QtGui import QGuiApplication,QIcon,QImage,QPixmap
from PyQt6 import QtCore, QtWidgets
from PyQt6.QtCore import Qt,QPropertyAnimation

import sys
from PyQt6.uic import loadUi
#import recursos

import recursos_iconos
import cv2

from output import *
from library_new.tesis_maestri import *

from PyQt6.QtWidgets import QApplication, QMainWindow, QPushButton, QFileDialog

import os

import torch
import torchvision
from torchvision import transforms as torchtrans  
import torchvision.transforms as transforms

import matplotlib.pyplot as plt
import numpy as np
import matplotlib.patches as patches
from PIL import Image

import sqlite3

"""/// Listado de variable  utilizadas y funciones\\
    self.ruta_variable => Variable que guarda la dir de folder
    self.list_image    => Variable que contiene la lista de imagenes
    self.imgproyectada => variable para determinar que imagen se esta visualizando
    self.pag = dice en que pagina esta y reinicia
    timage= saber si es la imagen original o la procesada
    """


class mainUI(QMainWindow):
    def __init__(self):
        super(mainUI,self).__init__()
        loadUi('Main_GUI.ui',self)
        self.ruta_carpeta=None
        self.list_images=[]
        self.imgproyectada=-1
        self.pag=0
        self.timage=0
        self.b_home.clicked.connect(lambda: self.pages.setCurrentWidget(self.p_home))	 #botones para cambiar de pagina
        self.b_model1.clicked.connect(lambda: self.pages.setCurrentWidget(self.p_model1))
        self.b_history.clicked.connect(lambda: self.pages.setCurrentWidget(self.p_history))
        self.b_history.clicked.connect(self.historial)
        self.b_model1.clicked.connect(self.identificador)
        self.b_model2.clicked.connect(lambda: self.pages.setCurrentWidget(self.p_model2))
        self.b_model2.clicked.connect(self.identificador2)
        self.b1_select_3.clicked.connect(self.leer_direc)  # Boton para seleccionar la carpeta
        self.b1_select_2.clicked.connect(self.leer_direc)
        self.b3_left_3.clicked.connect(self.pasarimage)   #boton izquierda para pasar datos
        self.b4_right_3.clicked.connect(self.pasarimage2)
        self.b3_left_2.clicked.connect(self.pasarimage)   #boton izquierda para pasar datos
        self.b4_right_2.clicked.connect(self.pasarimage2)

        # Botones para 
        self.b5_csv_2.clicked.connect(self.downloadcsv)
        self.b5_csv_2.clicked.connect(self.downloadshape)
        #borrado de base de dato
        self.b_borrarbase.clicked.connect(self.borrartodo)

        #programando si es original o procesada para muestra de imagen
        self.b_original.clicked.connect(self.original)
        self.b_process.clicked.connect(self.procesada)

        self.b2_process_2.clicked.connect(self.procesar_detection)
        # modelo de deteccion
        self.modeld = torch.load(("./library_new/modelo/content"),map_location=torch.device('cpu'))
        self.modeld.eval()

        
    
    def original(self):
        self.timage=0
        self.proyect_image()
    def procesada(self):
        # falta comprobacion si ya procesaron :D
        self.timage=1
        self.proyect_image()

    def identificador(self):
        if (self.pag):
            if (self.ruta_carpeta):
                self.pag=0
                self.imgproyectada=0    
                self.proyect_image()

    def identificador2(self):
        if (not(self.pag)):
            
            if (self.ruta_carpeta):
                self.pag=1
                self.imgproyectada=0
                self.proyect_image()

    def leer_direc(self):
        self.ruta_carpeta = QFileDialog.getExistingDirectory(self,"Select Folder")
        if (self.ruta_carpeta):
            self.pruebafun()
    
    
    def pruebafun(self):
        if (self.ruta_carpeta):
            con=0
            self.list_images,con=ver_formato(self.ruta_carpeta)
            if (con>0):
                mensaje = "files uploaded successfully"
                QMessageBox.information(self, "information", mensaje)
                self.imgproyectada=0
                self.proyect_image()
                
            else:
                mensaje = "The folder does not contain the Phantom 4 format"
                QMessageBox.critical(self, "Error", mensaje)
        else:
            mensaje = "Folder not found"
            QMessageBox.critical(self, "Error", mensaje)
        
    def proyect_image(self):

        if (self.timage):

            self.image=imagen_etiquetada(self.ruta_carpeta,self.list_images[int(self.imgproyectada)])
           # Aqui debe proyectarse la imagen con las etiquetas de la base de datos
            
        else: 
            self.image=cv2.imread(self.ruta_carpeta+'/'+self.list_images[int(self.imgproyectada)],1)
            self.image=cv2.cvtColor(self.image,cv2.COLOR_BGR2RGB)
            
        #self.image=cv2.imread(self.ruta_carpeta+'/'+self.list_images[int(self.imgproyectada)],1)
        self.actualizartabla1()
        self.image= cv2.resize(self.image, (520, 414), interpolation=cv2.INTER_LINEAR)
        qformat=QImage.Format.Format_BGR888
        img = QImage(self.image,self.image.shape[1],
                        self.image.shape[0],
                        self.image.strides[0],qformat)
        img= img.rgbSwapped()
        if (self.pag):
            self.l_image_3.setPixmap(QPixmap.fromImage(img))
            
        else:
            self.l_image_2.setPixmap(QPixmap.fromImage(img))
            
            
        
    def pasarimage(self):
        
        if self.imgproyectada==0:
            self.imgproyectada=int(len(self.list_images))-1
        else:
            self.imgproyectada-=1
        self.proyect_image()

    def pasarimage2(self):
        
        if self.imgproyectada==int(len(self.list_images))-1:
            self.imgproyectada=0
        else:
            self.imgproyectada+=1
        self.proyect_image()
    
    def procesar_detection(self):
        if (self.ruta_carpeta):    
            #necesitamos direccion para ller la variable
        
            prediccion(self.ruta_carpeta,self.modeld,self.list_images)
            self.timage=1
            self.proyect_image()
            self.llenartabla2()
            mensaje = "complete"
            QMessageBox.information(self, "information", mensaje)

        else:
            mensaje = "the folder has not been selected"
            QMessageBox.critical(self, "Error", mensaje)


    def historial(self):
        print("hola")
    
    #funcion para borrar las tablas de bdd
    def borrartodo(self):
        conexion = sqlite3.connect('./library_new/test.db')
        cursor = conexion.cursor()
        # Borrar todos los datos de la tabla "registro_carpeta"
        cursor.execute("DELETE FROM registro_carpeta")
        # Borrar todos los datos de la tabla "tabla_imagenes"
        cursor.execute("DELETE FROM tabla_imagenes")
        # Borrar todos los datos de la tabla "resultado_imagen"
        cursor.execute("DELETE FROM resultado_imagen")
        # Confirmar los cambios
        conexion.commit()
        conexion.close()

    def llenartabla2(self):
        
        dic=consulta_tablas1(self.ruta_carpeta)
        
        self.tabla_d2.setRowCount(len(dic))
        for indice,imagenes in enumerate(dic):
            self.tabla_d2.setItem(indice,0,QtWidgets.QTableWidgetItem(str(imagenes["nombre"])))
            self.tabla_d2.setItem(indice,1,QtWidgets.QTableWidgetItem(str(imagenes["n_detection"])))
    
    def actualizartabla1(self):
        if (self.timage):
            dic2=actualizar_tabla2(self.ruta_carpeta,self.list_images[int(self.imgproyectada)])
        else:
            dic2=[]
        self.tabla_r.setRowCount(len(dic2))
        for indice,imagenes in enumerate(dic2):
            
            self.tabla_r.setItem(indice,0,QtWidgets.QTableWidgetItem(str(imagenes["pixel_min"])))
            self.tabla_r.setItem(indice,1,QtWidgets.QTableWidgetItem(str(imagenes["pixel_max"])))
            self.tabla_r.setItem(indice,2,QtWidgets.QTableWidgetItem(str(imagenes["lat"])))
            self.tabla_r.setItem(indice,3,QtWidgets.QTableWidgetItem(str(imagenes["long"])))

    def downloadshape(self):
        print("hola1")
        convertir_a_shapefile(self.ruta_carpeta)
    def downloadcsv(self):
        print("hola2")
        enumerar_en_csv(self.ruta_carpeta)
if __name__ == '__main__':
    app = QApplication(sys.argv)
    ui = mainUI()
    ui.show()
    app.exec()



"""  # Para llamar desde el otro codigo que genera automaticamente puic6
 class MiApp(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        self.ui = Ui_MainWindow() 
        self.ui.setupUi(self)

if __name__ == "__main__":
     app = QtWidgets.QApplication(sys.argv)
     mi_app = MiApp()
     mi_app.show()
     app.exec() """