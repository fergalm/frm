from ipdb import set_trace as idebug
import matplotlib.pyplot as plt
from pprint import pprint
import pandas as pd
import numpy as np
import re


import PyQt5.QtWidgets as QtWidget
import PyQt5.QtCore as QtCore

from checkablecombo import CheckableComboBox


"""
Abstract and concreate column filter classes. 

These are QWidget objects that also encode the logic of which rows should be filtered in and out 
"""


class AbstractColumnFilter(QtWidget.QWidget):
    changed = QtCore.Signal()

    def __init__(self, parent=None):
        QtWidget.QWidget.__init__(self, parent)

    def getFilteredIn(self):
        pass

    def reset(self):
        pass

    def onChange(self):
        print(f"Change detected in {self.col}: {self}")
        self.changed.emit()

    def show(self):
        self.setVisible(True)

    def hide(self) -> None:
        self.setVisible(False)


class CategoricalFilter(AbstractColumnFilter):
    def __init__(self, col, parent=None):
        AbstractColumnFilter.__init__(self, parent)

        self.label = QtWidget.QLabel(col.name)

        self.col = col
        self.idx = np.ones(len(col), dtype=bool)
        self.items = set(col)
        self.combo = CheckableComboBox()
        self.combo.addItems(self.items)
        self.combo.model().dataChanged.connect(self.onChange)

        layout = QtWidget.QVBoxLayout()
        layout.addWidget(self.label)
        layout.addWidget(self.combo)
        self.setLayout(layout)


    def getFilteredIn(self) -> np.ndarray:
        print('Debug GFI', self.col.name, np.sum(self.idx))
        return self.idx.copy()

    def onChange(self):
        print("Getting selection for CatFilter for ", self.col.name)
        selected = self.combo.getSelectedItems()
        print(selected)
        print(self.col.astype(str).values[:10])
        print(self.col.astype(str).values[-10:])
        idx = self.col.astype(str).isin(selected)

        print( np.all(self.col.astype(str) == '0'))
        print(np.sum(self.col.astype(str) == '0'))
        print(np.sum(self.col.astype(str) == '1'))
        print(np.where(idx == False))
        print(idx)

        #import pdb; pdb.set_trace()
        # print(idx[:10])
        try:
            self.idx = idx.values
        except AttributeError:
            self.idx = idx
        print('Debug', self.col.name, np.sum(idx))
        self.changed.emit()

    def setWidth(self, width):
        self.combo.setMaximumWidth(width)



class NumericFilter(AbstractColumnFilter):
    def __init__(self, col, parent=None):
        AbstractColumnFilter.__init__(self, parent)

        self.label = QtWidget.QLabel(col.name)

        self.col = col
        self.idx = np.ones(len(self.col), dtype=bool)
        self.edit = QtWidget.QLineEdit()
        self.edit.textChanged.connect(self.onChange)

        layout = QtWidget.QVBoxLayout()
        layout.addWidget(self.label)
        layout.addWidget(self.edit)
        self.setLayout(layout)
        self.show()

    def onChange(self):
        print(f"Change detected in {self.col}: {self}")
        text = self.edit.text()
        cmd = self.parseText(text)

        if cmd == "":
            self.idx = np.ones(len(self.col), dtype=bool)   # No filter
            self.changed.emit()
            return

        try:
            # This, of course, hideously insecure
            idx = eval(cmd)
            print("command was parsed")
        except SyntaxError:
            print("Command failed to parse")
            return

        try:
            idx = idx.values
        except AttributeError:
            pass

        print(idx)
        if not isinstance(idx, np.ndarray):
            return
        print("idx is numpy array")

        if len(idx) != len(self.col):
            return
        print("idx is correct length")
        self.idx = idx
        print("Emiting")
        self.changed.emit()


    def parseText(self, text):
        if text == "":
            return text

        operators = "<= >= == != < >".split()
        for op in operators:
            text = re.subn(op, f"self.col {op}", text)[0]
            print( text)
        return text

    def getFilteredIn(self):
        try:
            return self.idx.values.copy()
        except AttributeError:
            return self.idx.copy()

    def setWidth(self, width):
        self.edit.setMaximumWidth(width)


class StringFilter(AbstractColumnFilter):
    def __init__(self, col, parent=None):
        AbstractColumnFilter.__init__(self, parent)
        self.label = QtWidget.QLabel(col.name)

        self.col = col
        self.idx = np.ones(len(self.col), dtype=bool)
        self.edit = QtWidget.QLineEdit()
        self.edit.textChanged.connect(self.onChange)

        layout = QtWidget.QVBoxLayout()
        layout.addWidget(self.label)
        layout.addWidget(self.edit)
        self.setLayout(layout)
        self.show()

    def onChange(self):
        print(f"Change detected in {self.col}: {self}")

        text = self.edit.text()
        num_char = len(text)
        self.idx = text == self.col[:num_char]
        print(f"{np.sum(self.idx)} of {len(self.idx)} are true")
        self.changed.emit()

    def getFilteredIn(self):
        try:
            return self.idx.values.copy()
        except AttributeError:
            return self.idx.copy()

    def setWidth(self, width):
        self.edit.setMaximumWidth(width)


class FilterCollection(QtWidget.QWidget):
    changed = QtCore.Signal()

    def __init__(self, filter_list, parent=None):
        QtWidget.QWidget.__init__(self, parent)
        self.filter_list = filter_list

        self.layout = QtWidget.QHBoxLayout()
        for f in filter_list:
            self.layout.addWidget(f)
            #Whenever a filter is changed, the collection issues an "I have changed" signal
            f.changed.connect(self.onChange)
        self.setLayout(self.layout)
        self.show()

    def addFilter(self, col_filter):
        self.layout.addWidget(col_filter)

    def onChange(self):
        print("Filter collection is emiting a signal")
        self.changed.emit()

    def getFilteredIn(self) -> np.ndarray:

        f0 = self.layout.itemAt(0).widget()
        idx = f0.getFilteredIn()  #Get length of index array
        idx |= True

        print("   ---    ")
        # print(f0.col.name,  np.sum(idx), " of ", len(f0.col))
        for i in range(0, self.layout.count()):
            f = self.layout.itemAt(i).widget()
            idx2 = f.getFilteredIn()
            # print(f"anding with {idx2}, {type(idx2)}")
            print(f.col.name, np.sum(idx2), " of ", len(idx2))
            print("Idx is now:", np.sum(idx))

            idx &= idx2
        return idx

    def showColumn(self, i):
        self.filter_list[i].show()

    def hideColumn(self, i):
        self.filter_list[i].hide()