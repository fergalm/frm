from ipdb import set_trace as idebug
import matplotlib.pyplot as plt
from pprint import pprint
import pandas as pd
import numpy as np

import PyQt5.QtWidgets as QtWidget
import PyQt5.QtCore as QtCore
import PyQt5.QtGui as QtGui
import PyQt5.QtWidgets


class TableWidget(PyQt5.QtWidgets.QTableWidget):
    def __init__(self, df, num=1000, parent=None):
        self.num = num
        self.include_row = np.ones(len(self.df), dtype=bool)

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

    def drawFiltered(self, idx):
        self.setFilter(idx)
        self.draw()

    def setFilter(self, idx):
        assert len(idx) == len(self.df)
        self.include_row = idx

    def draw(self):
        # idebug()
        df = self.df
        df = df[self.include_row]
        df = df[:self.num]  #Max number of rows to display
        self.set_table_elements(df)

    def set_table_elements(self, df):
        self.resetTable()
        self.setRowCount(len(df))
        self.setHorizontalHeaderLabels(df.columns)

        QItem = PyQt5.QtWidgets.QTableWidgetItem  #Mnumonic
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
                item.setFlags( item.flags() & ~QtCore.Qt.EditRole)
                self.setItem(j, i, item)

        self.resizeColumnsToContents()
        self.resizeRowsToContents()

    def resetTable(self):
        """Clear data from the table"""
        row_count = self.rowCount()
        self.setRowCount(0)

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
