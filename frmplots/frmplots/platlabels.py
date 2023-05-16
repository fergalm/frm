"""

This will be a code to annotate a figure in such a way that labels don't overlap

It's not there yet
"""

import matplotlib.pyplot as plt 
import matplotlib as mpl
import numpy as np
import itertools 

"""
if is_visible[j]:

    for h in [left, center, right]:
        t.set_horizontal_alignment(h)
        for v in ["baseline","bottom",  "top"]
            t.set_vertical_alignment(v)
            is_overlap = self.overlap(t, self.tList[j])
            if not is_overlap:
                #Break out of the loop, the label goes here"
            
"""


class PlatLabel():
    """Hide some overlapping labels on a plot

    Given a set of matplotlib.Text objects, figure out which
    ones overlap, and which ones should be hidden to avoid
    overlaps.

    Usage:
    ml= MapLabel()

    for i in range(10)
        handle = mp.text(i, i, "Some text")
        ml.add(handle)

    #At this point, all labels are shown, whether they overlap
    #or not. This next method call hides those that overlap
    ml.render()


    Note:
    ----------
    MapLabel objects can be combined with the + operator.
    mapLabel1 + mapLabel2 returns a new mapLabel object
    that contains handles to the text objects in both original
    objects. This can be convenient to collect text handles
    from functions before calling render once on the combined
    set.
    """

    def __init__(self):
        self.tList = []


    def __add__(self, mapLabel2):
        tList = []
        tList.extend(self.tList)

        try:
            tList.extend(mapLabel2.tList)
        except AttributeError:
            raise NotImplementedError()

        out = PlatLabel()
        out.tList = tList
        return out


    def __iadd__(self, mapLabel2):
        try:
            self.tList.extend( mapLabel2.tList)
        except AttributeError:
            raise NotImplementedError()

        return self


    def add(self, t, level):
        """Add a text object to the class object.

        Inputs:
        -----------
        t
            (A matplotlib.Text object) as returned by
            mp.text()

        level
            (int or float) A priority number. If two text labels
            overlap, the one with the lower level is plotted first.

        Returns:
        ------------
        **None**

        """

        assert(isinstance(t, mpl.text.Text))
        t.level = level
        self.tList.append(t)

    def text(self, x, y, text, **kwargs):
        level = kwargs.pop('level', 1)
        self.add(plt.text(x, y, text, **kwargs), level)

    def render(self):
        """Compute which labels overlap and which ones to hide

        Returns:
        ------------
        **None**
        """
        levels = np.array( list(map(lambda x: x.level, self.tList)))

        srt = np.argsort(levels)
        is_visible = np.ones_like(srt, dtype=bool)

        for i in range(len(srt)):
            t_i = self.tList[ srt[i] ]
            t_i.set_visible(True)

            j = 0
            while j < i:
                t_j = self.tList[ srt[j] ]
                if self.overlap(t_i, t_j) and is_visible[j]:
                    # print(f"{i} overlaps with {j}, {t_j}")
                    # print(f"Marking elt {i} ({t_i}) as invisible")
                    is_visible[i] = False 
                    t_i.set_visible(False)
                    break
                j+= 1


    # def advanced_render(self):

    #     #Level 1 comes before level 2 etc.
    #     tList = sorted(self.tList, key=lambda x: x.level)
    #     nLabel = len(tList)

    #     ha = "left right".split()
    #     va = "bottom top".split()
    #     placement_options = list(itertools.product(va, ha))

    #     for i in range(nLabel):
    #         t_i = tList[i]
    #         t_i.set_visible(False)

    #     for i in range(nLabel):
    #         t_i = tList[i]
    #         t_i.set_visible(True)

    #         for pl in placement_options:
    #             t_i.set_horizontalalignment(pl[1])
    #             t_i.set_verticalalignment(pl[0])
    #             print(t_i.get_text(), pl, t_i.get_ha(), t_i.get_va())
    #             print( getTextBbox(t_i))

    #             flag = True 
    #             for j in range(i):
    #                 t_j = tList[j]

    #                 if  t_j.get_visible() and self.overlap(t_i, t_j):
    #                     flag = False 
    #                     #We found an overlap, this placement is invalid, stop looking
    #                     #Try the next placement option
    #                     break 

    #             plt.pause(.1)
    #             print(t_i)
    #             import ipdb; ipdb.set_trace()

    #             if flag:
    #                 #We found a legimiate placement, stop looking for more
    #                 #Try to place the label i+1
    #                 t_i.set_visible(True)
    #                 break
    #             t_i.set_visible(False)


    def overlap(self, t1, t2):
        """Do two text objects overlap?

        Internal Function.

        Inputs:
        -------------
        t1, t2
            (matplotlib.Text objects)


        Returns:
        ------------
        boolean
        """
        c1 = getTextBbox(t1)
        c2 = getTextBbox(t2)
        return c1.intersection(c1,c2) is not None


    # def drawAllBboxes(self):
    #     for t in self.tList:
    #         self.drawBBox(t)

    # def drawBbox(self, t, color='k'):
    #     bbox = getTextBbox(t)
    #     corners = bbox.corners()
    #     print(corners)
    #     for i in range(0, len(corners)):
    #         x0, y0 = corners[i-1]
    #         x1, y1 = corners[i]
    #         plt.plot([x0, x1], [y0,y1], '-', color=color, lw=1)


def getTextBbox(text):
    """

    Based on Text.get_window_extent()

    This is called out as a separate function because I expect it will be much more work
    in other plotting toolkits

    """
    return text.get_window_extent()


def getPointsPerDataUnit():
    """Compute number of data units per font point size
    
    Useful if you want to draw text at a particular size relative your data points.
    Naturally the scaled value is invalidated once the axis limits are changed
    or the figure size is changed.

    Returns
    -------
    x_points_per_data, y_points_per_data
        Number of font points per data unit in x and y. 

    
    Example
    --------
    To create text that is as high as 10 data points in the y-axis Do::

        xs, ys = get_points_per_data_unit()
        plt.text(0, 10, "Text", fontsize=10*ys)

    The top of the T in text will align with the "20" position on the y axis

    """
    bbox = plt.gca().get_window_extent().get_points()
    fig = plt.gcf()
    ax = plt.gca()

    bbox = ax.get_window_extent().transformed(fig.dpi_scale_trans.inverted())
    width_inches, height_inches = bbox.width, bbox.height
    width_points = width_inches * fig.dpi
    height_points = height_inches * fig.dpi

    axl = plt.axis()
    yscale_data = axl[3] - axl[2] 
    xscale_data = axl[1] - axl[0]

    #This isn't a hard requirement, I just haven't thought through
    #the consequences of -ve values yet.
    # assert np.all(np.array([yscale_points, xscale_points, yscale_data, xscale_data]) > 0)

    x_points_per_data = width_points / xscale_data 
    y_points_per_data = height_points / yscale_data * 1
    return x_points_per_data, y_points_per_data


def test1():
    plt.clf()
    plt.axis([-1, 1, -1, 1])
    pl = PlatLabel()
    pl.add(plt.text(0, 0, "Text 0"), 1)
    pl.add(plt.text(.1, 0, "Text 1"), 2)

    assert pl.tList[1].get_visible()
    pl.render()
    assert not pl.tList[1].get_visible()


def test2():
    state = np.random.RandomState(1234)
    plt.clf()
    # plt.axis([0, 1, 0, 1])

    pl = PlatLabel()

    num = 100
    state.rand(10)
    x = state.rand(num)
    y = state.rand(num)
    for i in range(num):
        pl.text(x[i], y[i], f"This is the number {i}", clip_on=True, level=int(i/10))
    pl.render()