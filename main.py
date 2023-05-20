
from PyQt6.QtWidgets import QMainWindow, QApplication,QLineEdit,QMessageBox

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
"""/// Listado de variable  utilizadas y funciones\\
    self.ruta_variable => Variable que guarda la dir de folder
    self.list_image    => Variable que contiene la lista de imagenes
    self.imgproyectada => variable para determinar que imagen se esta visualizando
    self.pag = dice en que pagina esta y reinicia
    """


class mainUI(QMainWindow):
    def __init__(self):
        super(mainUI,self).__init__()
        loadUi('Main_GUI.ui',self)
        self.ruta_carpeta=None
        self.list_images=[]
        self.imgproyectada=0
        self.pag=0
        self.b_home.clicked.connect(lambda: self.pages.setCurrentWidget(self.p_home))	 #botones para cambiar de pagina
        self.b_model1.clicked.connect(lambda: self.pages.setCurrentWidget(self.p_model1))
        self.b_model1.clicked.connect(self.identificador)
        self.b_model2.clicked.connect(lambda: self.pages.setCurrentWidget(self.p_model2))
        self.b_model2.clicked.connect(self.identificador2)
        self.b1_select_3.clicked.connect(self.leer_direc)  # Boton para seleccionar la carpeta
        self.b1_select_2.clicked.connect(self.leer_direc)
        self.b3_left_3.clicked.connect(self.pasarimage)   #boton izquierda para pasar datos
        self.b4_right_3.clicked.connect(self.pasarimage2)
        self.b3_left_2.clicked.connect(self.pasarimage)   #boton izquierda para pasar datos
        self.b4_right_2.clicked.connect(self.pasarimage2)
    
    def identificador(self):
        if (self.pag):
            self.pag=0
            self.imgproyectada=0
            self.proyect_image()

    def identificador2(self):
        if (not(self.pag)):
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
            
    def proyect_image(self,):
        
        self.image=cv2.imread(self.ruta_carpeta+'/'+self.list_images[int(self.imgproyectada)],1)
        self.image=cv2.cvtColor(self.image,cv2.COLOR_BGR2RGB)
        self.image= cv2.resize(self.image, (520, 414), interpolation=cv2.INTER_LINEAR)
        qformat=QImage.Format.Format_BGR888
        img = QImage(self.image,self.image.shape[1],
                        self.image.shape[0],
                        self.image.strides[0],qformat)
        img= img.rgbSwapped()
        if (self.pag):
            self.l_image_2.setPixmap(QPixmap.fromImage(img))
        else:
            self.l_image_3.setPixmap(QPixmap.fromImage(img))
        
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