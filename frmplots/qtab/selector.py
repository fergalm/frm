

from ipdb import set_trace as idebug

import PyQt5.QtWidgets as QtWidget
import PyQt5.QtWidgets


class ColumnSelector(PyQt5.QtWidgets.QDialog):
    def __init__(self, table, parent=None):
        self.table = table
        df = table.df
        PyQt5.QtWidgets.QDialog.__init__(self, parent)
        self.setMinimumWidth(150)

        layout = PyQt5.QtWidgets.QVBoxLayout(self)
        button1 = QtWidget.QPushButton("Select All")
        button1.clicked.connect(self.showAll)
        button2 = QtWidget.QPushButton("Hide All")
        button2.clicked.connect(self.hideAll)
        layout.addWidget(button1)
        layout.addWidget(button2)
        self.setWindowTitle("Select Columns")

        self.boxes = []
        for i, col in enumerate(df.columns):
            print(i, col)
            checkbox = QtWidget.QCheckBox(col)
            checkbox.setChecked(True)
            checkbox.stateChanged.connect(self.onToggle)
            layout.addWidget(checkbox)
            self.boxes.append(checkbox)
        self.layout = layout
        self.show()

        self.keyReleaseEvent = self.quit

    def quit(self, eventQKeyEvent):
        key = eventQKeyEvent.key()
        if key == 81:  #The letter [q]
            self.hide()

    def showAll(self):
        self.table.showAll()  # Maybe unecessary?

        # Update all the checkboxes
        for box in self.boxes:
            box.setChecked(True)

    def hideAll(self):
        self.table.hideAll()  # Maybe unecessary?

        # Update all the checkboxes
        for box in self.boxes:
            box.setChecked(False)

    def onToggle(self, state):
        sender = self.sender()
        self.table.toggleColumn(sender.text(), state > 0)
