try:
    import PyQt6.QtWidgets as QtWidget
    import PyQt6.QtCore as QtCore
    import PyQt6.QtGui as QtGui
except ImportError:
    import PyQt5.QtWidgets as QtWidget
    import PyQt5.QtCore as QtCore
    import PyQt5.QtGui as QtGui


from ipdb import set_trace as idebug
import numpy as np



# from checkablecombo import CheckableComboBox


"""
Abstract and concrete column filter classes. 

These are QWidget objects that also encode the logic of which rows should be filtered in and out 
"""

    
class AbstractColumnFilter(QtWidget.QWidget):
    changed = QtCore.Signal()

    def __init__(self, col, parent=None):
        QtWidget.QWidget.__init__(self, parent)
        self.keyReleaseEvent = self.onChange
        self.col = col 
    
    def __repr__(self):
        return f"<{type(self)} on column {self.col}>"
    
    def onChange(self, keyPress):
        self.changed.emit()
    
    def validate(self, df, col):
        pass 
    
    def applyFilter(self, df, col):
        pass 
    
    def show(self):
        self.setVisible(True)

    def hide(self) -> None:
        self.setVisible(False)


class StringFilter(AbstractColumnFilter):
    def __init__(self, col, parent=None):
        AbstractColumnFilter.__init__(self, col, parent)

        self.label = QtWidget.QLabel(col)

        self.edit = QtWidget.QLineEdit()
        self.edit.textChanged.connect(self.onChange)

        layout = QtWidget.QVBoxLayout()
        layout.addWidget(self.label)
        layout.addWidget(self.edit)
        self.setLayout(layout)
        self.show()

    def validate(self, df, col):
        #Any column can always be treated as text 
        return True 
    
    def applyFilter(self, df):
        text = self.edit.text()
        num_char = len(text)

        if num_char == 0:
            return df 
        
        idx = text == df[self.col].astype(str).str[:num_char]
        if np.any(idx):
            return df[idx].copy()
        return  df[idx]

class NumericFilter(AbstractColumnFilter):
    def __init__(self, col, parent=None):
        AbstractColumnFilter.__init__(self, col, parent)

        self.label = QtWidget.QLabel(col)

        self.edit = QtWidget.QLineEdit()
        self.edit.textChanged.connect(self.onChange)

        layout = QtWidget.QVBoxLayout()
        layout.addWidget(self.label)
        layout.addWidget(self.edit)
        self.setLayout(layout)
        self.show()

    def validate(self, df, col):
        try:
            df[col].astype(float)
            return True 
        except ValueError:
            return False 

    def applyFilter(self, df):
        text = self.edit.text()
        
        if text == "":
            return df 
        
        cmd = f"df[self.col] {text}"
        
        try:
            # This, of course, hideously insecure
            idx = eval(cmd)
        except SyntaxError:
            print(f"Command failed to parse: {cmd}")
            return df
        return df[idx].copy()



class DatetimeFilter(AbstractColumnFilter):
    def __init__(self, col, parent=None):
        AbstractColumnFilter.__init__(self, col, parent)

        self.label = QtWidget.QLabel(col)

        self.edit = QtWidget.QLineEdit()
        self.edit.textChanged.connect(self.onChange)

        layout = QtWidget.QVBoxLayout()
        layout.addWidget(self.label)
        layout.addWidget(self.edit)
        self.setLayout(layout)
        self.show()

    def validate(self, df, col):
        try:
            pd.to_datetime(df[col])
            return True 
        except ValueError:
            print("Failed to validate as a datetime")
            return False 

    def applyFilter(self, df):
        text = self.edit.text()
        
        if text == "":
            return df 
        
        cmd = f"df[self.col] {text}"
        
        try:
            # This, of course, hideously insecure
            idx = eval(cmd)
        except SyntaxError:
            print(f"Command failed to parse: {cmd}")
            return df
        return df[idx].copy()

