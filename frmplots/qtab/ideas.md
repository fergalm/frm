
# Idea

## Filters
Each column of a QT table should have a filter option. Filters include
* A subset filter (only 100 rows, starting from row 200)
* A categorical filter (if there are fewer than, say 10 values)
* A numeric filter  [ >5 & < 10]
* A string filter (start of string matches, maybe regexes?)
* A separate date filter?

Numeric filter should include Nan filter


```
class ColFiter(QWidget):
    def getFilteredIn(self): -> idx
        ...

    def reset(self):
        ...

    @signal
    def onChange(self):
        ...


```
## Plots
Each column of a QT table should have a histogram on top of it showing the range of values (if it's numeric).
Different colour indicate selected/unselected by the filter, or selected by this filter, but unselected by another filter



