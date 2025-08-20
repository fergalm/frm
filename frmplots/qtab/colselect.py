try:
    import PyQt6.QtWidgets as QtWidget
    import PyQt6.QtCore as QtCore
    import PyQt6.QtGui as QtGui
    import PyQt6.QtWidgets
except ImportError:
    import PyQt5.QtWidgets as QtWidget
    import PyQt5.QtCore as QtCore
    import PyQt5.QtGui as QtGui
    import PyQt5.QtWidgets


class ColumnSelector(QtWidget.QDialog):
    changed = QtCore.Signal()
    
    def __init__(self, columns, parent=None):
        # self.table = table
        # df = table.df
        QtWidget.QDialog.__init__(self, parent)
        self.setMinimumWidth(150)

        layout = QtWidget.QVBoxLayout(self)
        button1 = QtWidget.QPushButton("Select All")
        button1.clicked.connect(self.showAll)
        button2 = QtWidget.QPushButton("Hide All")
        button2.clicked.connect(self.hideAll)
        layout.addWidget(button1)
        layout.addWidget(button2)
        self.setWindowTitle("Select Columns")

        self.boxes = []
        for i, col in enumerate(columns):
            print(i, col)
            checkbox = QtWidget.QCheckBox(col)
            checkbox.setChecked(True)
            checkbox.stateChanged.connect(self.onToggle)
            layout.addWidget(checkbox)
            self.boxes.append(checkbox)
        self.layout = layout
        self.hide()

        self.keyReleaseEvent = self.quit

    def quit(self, eventQKeyEvent):
        key = eventQKeyEvent.key()
        if key == 81:  #The letter [q]
            self.hide()

    def toggleVisible(self):
        if self.isVisible():
            self.hide()
        else:
            self.show()
            
    def showAll(self):
        # Update all the checkboxes
        for box in self.boxes:
            box.setChecked(True)
        self.changed.emit()
         
    def hideAll(self): 
        # Update all the checkboxes
        for box in self.boxes:
            box.setChecked(False)
        self.changed.emit()
    
    def onToggle(self, state):
        self.changed.emit()

    def getSelected(self):
        out = []
        for box in self.boxes:
            if box.isChecked():
                out.append(box.text())
        return out 
