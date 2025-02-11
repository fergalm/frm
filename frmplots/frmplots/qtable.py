# from ipdb import set_trace as idebug

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

"""
A quick widget that can be called from ipython to display a dataframe
"""

class QTable(QtWidget.QDialog):
    def __init__(self, df, num=1000, title="Dataframe", parent=None):
        super().__init__(parent)
        app = QtWidget.QApplication.instance()
        if app is None:
            app = QtWidget.QApplication([])

        self.create_layout(df, num, title)
        self.selector = ColumnSelector(self.table)
        self.selector.hide()

        #I think I have to subclass QTableWidget and override keyReleaseEvent
        self.keyReleaseEvent = self.process_key_press
        self.show()

    def create_layout(self, df, num, title):
        self.button = QtWidget.QPushButton("Show/hide Columns")
        self.button.clicked.connect(self.toggle_button)
        self.table = TableWidget(df, num=num)


        layout = QtWidget.QVBoxLayout()
        layout.addWidget(self.button)
        layout.addWidget(self.table)
        self.setLayout(layout)

        self.resize(self.table.width(), self.table.height())
        self.setMaximumSize(self.table.getMaxWidth(), 10000)

        self.setWindowTitle(title)

    def process_key_press(self, eventQKeyEvent):
        key = eventQKeyEvent.key()
        if key == 81:  #The letter [q]
            self.selector.hide()
            self.hide()
            self.close()

    def toggle_button(self):
        if self.selector.isVisible():
            self.selector.hide()
        else:
            self.selector.show()

    def toggleColumn(self, sender_label, state):
        cols = self.table.df.columns
        for i in range(self.ncol):
            if cols[i] == sender_label:
                if state:
                    self.table.showColumn(i)
                else:
                    self.table.hideColumn(i)

    def showAll(self):
        for i in range(self.ncol):
            self.table.showColumn(i)

    def hideAll(self):
        for i in range(self.ncol):
            self.table.hideColumn(i)


class TableWidget(QtWidget.QTableWidget):
    def __init__(self, df, num=1000, parent=None):
        self.df = df[:num]
        self.nrow = len(self.df)
        self.ncol = len(self.df.columns)
        super().__init__(self.nrow, self.ncol, parent=parent)
        self.max_width = 1000

        # PyQt5.QtWidgets.QTableWidget(self.nrow, self.ncol)
        self.set_table_elements(self.df)
        self.set_size_policy()
        self.draw_row_guides()
        self.setSortingEnabled(True)

        header = self.horizontalHeader()
        header.sectionClicked.connect(self.draw_row_guides)
        self.resizeRowsToContents()

    def set_table_elements(self, df):
        self.setHorizontalHeaderLabels(df.columns)

        QItem = QtWidget.QTableWidgetItem  #Mnumonic
        for i, key in enumerate(df.columns):
            col = df[key]
            for j, elt in enumerate(col):
                # print(i, j, elt)
                if elt is None:
                    item = QItem(" ")
                    continue
                try:
                    item = QItem("%g"% (elt))
                except TypeError:
                    item = QItem(str(elt))
                # Mark item as readonly
                #item.setFlags( item.flags() & ~QtCore.Qt.EditRole)
                item.setFlags( item.flags()) 
                self.setItem(j, i, item)

        self.resizeColumnsToContents()
        self.resizeRowsToContents()

    def set_size_policy(self):
        width_pix = self.horizontalHeader().length() + self.verticalHeader().width() + 20
        height_pix = self.verticalHeader().length() + self.horizontalHeader().width()
        self.setMaximumSize(width_pix, height_pix)
        self.max_width = width_pix

        width_pix = min(width_pix, 1000)
        height_pix = min(height_pix, 1000)
        self.resize(width_pix, height_pix)

    def getMaxWidth(self):
        return self.max_width

    def draw_row_guides(self):
        """This is called by the HeaderView.isClicked signal"""
        guideStep = 5

        for i in range(self.nrow):
            clr = QtGui.QColor('#FFFFFF')
            if i % guideStep == guideStep -1:
                clr = QtGui.QColor('#DDDDFF')

            for j in range(self.ncol):
                item = self.item(i, j)
                if item is not None:
                    item.setBackground(clr)

    def toggleColumn(self, sender_label, state):
        cols = self.df.columns
        for i in range(self.ncol):
            if cols[i] == sender_label:
                if state:
                    self.showColumn(i)
                else:
                    self.hideColumn(i)

    def showAll(self):
        for i in range(self.ncol):
            self.showColumn(i)

    def hideAll(self):
        for i in range(self.ncol):
            self.hideColumn(i)


class ColumnSelector(QtWidget.QDialog):
    def __init__(self, table, parent=None):
        self.table = table
        df = table.df
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


if __name__ == "__main__":
    import pandas as pd
    import sys

    if len(sys.argv) != 2:
        print("Usage: qtable.py file.csv")
        sys.exit(1)

    app = QtWidget.QApplication([])
    df = pd.read_csv(sys.argv[1])
    print("loading data")
    tab = QTable(df)
    tab.show()
    sys.exit(app.exec())
