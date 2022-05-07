import sys
from PyQt5 import QtWidgets, QtGui, QtCore
from PyQt5.QtCore    import pyqtSlot
from PyQt5.QtWidgets import QApplication, QWidget, QHBoxLayout, QVBoxLayout, QPushButton
from PyQt5.QtCore import *

class Ui_MainWindow(object):
    def setupUi(self, MainWindow):       
        MainWindow.setObjectName("MainWindow")       
        MainWindow.resize(1920, 1080)        
        self.centralWidget = QtWidgets.QWidget(MainWindow)           
        self.centralWidget.setObjectName("centralWidget")        
        self.pushButton = QtWidgets.QPushButton(self.centralWidget)
        self.pushButton.setGeometry(QtCore.QRect(600, 900, 90, 25))        
        self.pushButton.setObjectName("pushButton")        
        self.pushButton_2 = QtWidgets.QPushButton(self.centralWidget)
        self.pushButton_2.setGeometry(QtCore.QRect(130, 90, 90, 25))        
        icon = QtGui.QIcon()
        icon.addPixmap(QtGui.QPixmap("./1111.gif"), QtGui.QIcon.Normal, QtGui.QIcon.Off)        
        self.pushButton_2.setIcon(icon)        	
        self.pushButton_2.setObjectName("pushButton_2")        
        self.pushButton_3 = QtWidgets.QPushButton(self.centralWidget)
        self.pushButton_3.setGeometry(QtCore.QRect(90, 180, 75, 23))
        self.pushButton_3.setObjectName("pushButton_3")
        self.pushButton_3.setEnabled(False)        
        self.pushButton_4 = QtWidgets.QPushButton(self.centralWidget)
        self.pushButton_4.setGeometry(QtCore.QRect(190, 180, 75, 23))
        self.pushButton_4.setObjectName("pushButton_4")
        MainWindow.setCentralWidget(self.centralWidget)
        self.pushButton_2.setCheckable(True)
        self.pushButton.pressed.connect(self.ButtonState)
        self.pushButton_2.toggled.connect(self.Button2Proc)
        self.pushButton_4.clicked.connect(MainWindow.close)
        self.retranslateUi(MainWindow)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)
    def ButtonState(self):        
        if self.pushButton.isDown():            
            print(self.pushButton.text() + ' pressed')            
            self.pushButton_2.toggle()            
    def Button2Proc(self):         
        status =  "True" if self.pushButton_2.isChecked() else "False"
        print(self.pushButton_2.text() + " button status is " + status)

    def retranslateUi(self, MainWindow):       
        _translate = QtCore.QCoreApplication.translate 
        MainWindow.setWindowTitle(_translate("MainWindow", "MainWindow"))
        self.pushButton.setText(_translate("MainWindow", "Button1"))
        self.pushButton.setShortcut(_translate("MainWindow", "Alt+B"))
        self.pushButton_2.setText(_translate("MainWindow", "&Save"))
        self.pushButton_3.setText(_translate("MainWindow", "OK"))
        self.pushButton_4.setText(_translate("MainWindow", "Cancel"))





if __name__ == '__main__':
    app = QtWidgets.QApplication(sys.argv)    
    MainWindow = QtWidgets.QMainWindow()    
    ui = Ui_MainWindow()    
    ui.setupUi(MainWindow)    
    MainWindow.show()    
    sys.exit(app.exec_())

