import io, sys
from PyQt5 import QtCore, QtGui, QtWidgets, uic
from PyQt5.QtCore import Qt, QFile, QDataStream, QBuffer
from PyQt5.QtGui import QPen
from PyQt5.QtWidgets import (QWidget, QPushButton,
                             QHBoxLayout, QVBoxLayout, QApplication)
from PIL import Image
import PIL.ImageOps
import numpy as np

class MainWindow(QtWidgets.QMainWindow):
    def saveToFile(self):
        buffer = QBuffer()
        buffer.open(QBuffer.ReadWrite)
        qimg = self.label.pixmap().toImage() # QImage
        qimg.save(buffer, "PNG")
        pimg = Image.open(io.BytesIO(buffer.data())) # PIL Image
        pimg = pimg.resize((28, 28))
        pimg = pimg.convert("L") # v črno-belo
        pimg = PIL.ImageOps.invert(pimg) # negativ
        npimg = np.array(pimg) # numpy array

        self.img_array = npimg # numpy array for model

        npimg = np.insert(npimg, 0, 0) # vrednost 0 pomeni neznano število
        column_heads = ["pixel" + str(i) for i in range(784)]
        column_heads.insert(0, "label")

        np.savetxt("new_digit.csv", [column_heads], delimiter=",", fmt="%s")
        f = open("new_digit.csv", "ab")
        np.savetxt(f, [npimg], delimiter=",", fmt="%d")
        f.close()
        #print(pimg.save("new_digit.png","png"))

    def clear_canvas(self):
        self.canvas = QtGui.QPixmap(600, 600)
        self.canvas.fill(QtGui.QColor("black"))
        self.label.setPixmap(self.canvas)

    def __init__(self):
        super().__init__()


        self.label = QtWidgets.QLabel()
        self.pred_number = QtWidgets.QLabel()
        self.pred_number.setText("Number: ")

        self.canvas = QtGui.QPixmap(600, 600)  # velikost canvasa za risat 28 *2^3 # zdi se mi premalo zato povcal
        self.canvas.fill(QtGui.QColor("black"))  # zacetna barva canvasa
        #self.setFixedSize(224, 244)  # velikost windowa


        self.recoqnizebutton = QtWidgets.QPushButton("Recoqnize number")
        self.savebutton = QtWidgets.QPushButton(self)
        self.savebutton.setText("Save")
        self.savebutton.clicked.connect(self.saveToFile)  # save to file event
        self.label.setPixmap(self.canvas)

        self.clearbutton = QtWidgets.QPushButton("Clear") # button to clear canvas
        self.clearbutton.clicked.connect(self.clear_canvas)

        w = QtWidgets.QWidget()
        l = QtWidgets.QVBoxLayout()
        w.setLayout(l)
        l.addWidget(self.label)
        l.addWidget(self.pred_number)
        l.addWidget(self.recoqnizebutton)
        l.addWidget(self.savebutton)
        l.addWidget(self.clearbutton)

        self.setCentralWidget(w)

        self.last_x, self.last_y = None, None

    def mouseMoveEvent(self, e):
        napaka = -10
        if self.last_x is None: # First event.
            self.last_x = e.x() + napaka
            self.last_y = e.y()
            return # Ignore the first time.

        painter = QtGui.QPainter(self.label.pixmap())
        pen = QPen(Qt.white)
        pen.setWidth(20) #povecal iz 8 na 20
        painter.setPen(pen)
        painter.drawLine(self.last_x, self.last_y, e.x()+napaka, e.y())
        painter.end()
        self.update()

        # Update the origin for next time.
        self.last_x = e.x()+napaka
        self.last_y = e.y()

    def mouseReleaseEvent(self, e):
        self.last_x = None
        self.last_y = None


# import this in digit_rec_model.py to add model
'''
app = QtWidgets.QApplication(sys.argv)
window = MainWindow()
window.show()
app.exec_()
'''
