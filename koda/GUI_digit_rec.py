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
        npimg = np.insert(npimg, 0, 0) # vrednost 0 pomeni neznano število
        column_heads = ["pixel" + str(i) for i in range(784)]
        column_heads.insert(0, "label")

        np.savetxt("new_digit.csv", [column_heads], delimiter=",", fmt="%s")
        f = open("new_digit.csv", "ab")
        np.savetxt(f, [npimg], delimiter=",", fmt="%d")
        f.close()
        #print(pimg.save("new_digit.png","png"))

    def __init__(self):
        super().__init__()

        self.label = QtWidgets.QLabel()

        self.canvas = QtGui.QPixmap(224, 224)  # velikost canvasa za risat 28 *2^3
        self.canvas.fill(QtGui.QColor("white"))  # zacetna barva canvasa
        self.setFixedSize(224, 244)  # velikost windowa

        self.savebutton = QtWidgets.QPushButton(self)
        self.savebutton.setText("Save")
        self.savebutton.clicked.connect(self.saveToFile)  # save to file event
        self.label.setPixmap(self.canvas)


        w = QtWidgets.QWidget()
        l = QtWidgets.QVBoxLayout()
        w.setLayout(l)
        l.addWidget(self.label)
        l.addWidget(self.savebutton)

        self.setCentralWidget(w)

        self.last_x, self.last_y = None, None

    def mouseMoveEvent(self, e):
        napaka = -10
        if self.last_x is None: # First event.
            self.last_x = e.x() + napaka
            self.last_y = e.y()
            return # Ignore the first time.

        painter = QtGui.QPainter(self.label.pixmap())
        pen = QPen(Qt.black)
        pen.setWidth(8)
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


app = QtWidgets.QApplication(sys.argv)
window = MainWindow()
window.show()
app.exec_()