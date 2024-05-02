"""

This will be a code to annotate a figure in such a way that labels don't overlap

It's not there yet
"""

from ipdb import set_trace as idebug
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


class AbstractPlatLabel():
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
        # print(c1, c2, c1.intersection(c1,c2))
        return c1.intersection(c1,c2) is not None

    def render(self):
        """Compute which labels overlap and which ones to hide

        Returns:
        ------------
        **None**
        """
        raise NotImplementedError("Don't call abstract class directly")
    

class SimplePlatLabel(AbstractPlatLabel):
    """Put all the labels to the upper right of their points. 

    Hide lower priority labels that overlap
    """
    def render(self):
        levels = np.array( list(map(lambda x: x.level, self.tList)))

        srt = np.argsort(levels)
        print(srt)
        is_visible = np.ones_like(srt, dtype=bool)

        for i in range(len(srt)):
            t_i = self.tList[ srt[i] ]
            t_i.set_visible(True)

            j = 0
            while j < i:
                print(i, j)
                t_j = self.tList[ srt[j] ]
                if self.overlap(t_i, t_j) and is_visible[j]:
                    print(f"{srt[i]} overlaps with {srt[j]}, {t_j}")
                    print(f"Marking elt {srt[i]} ({t_i}) as invisible")
                    is_visible[i] = False
                    t_i.set_visible(False)
                    break
                j+= 1




class PlatLabel(AbstractPlatLabel):
    """Try different options for ha and va before hiding label"""

    def __init__(self):
        PlatLabel.__init__(self)

    def render(self):

        # Get priority order of labels
        tList = sorted(self.tList, key=lambda x: x.level)
        nLabel = len(tList)

        ha = "left center right".split()
        va = "bottom center top".split()
        placement_options = list(itertools.product(va, ha))

        #Hide all labels to start
        for i in range(nLabel):
            t_i = tList[i]
            t_i.set_visible(False)

        for i in range(nLabel):
            t_i = tList[i]
            #Show this label
            t_i.set_visible(True)

            if i == 0:
                continue 

            #See if we need to hide it again
            # if i == 1:
            #     idebug()
            for pl in placement_options:
                t_i.set_horizontalalignment(pl[1])
                t_i.set_verticalalignment(pl[0])
                # print(  t_i.get_text(), pl, t_i.get_ha(), t_i.get_va())
                t_i.set_visible(True)
                # plt.pause(1)
                # print( getTextBbox(t_i))
                # _x0 = getTextBbox(t_i).x0
                # assert _x0 > 0

                is_displayable = True

                for j in range(i):
                    t_j = tList[j]

                    if  t_j.get_visible() and self.overlap(t_i, t_j):
                        # print(f"{i} overlaps with {j}")
                        is_displayable = False
                        #We found an overlap, this placement is invalid, stop looking
                        #Try the next placement option
                        break

                if is_displayable:
                    #We found a legimiate placement, stop looking for more
                    #Try to place the label i+1
                    t_i.set_visible(True)
                    # plt.pause(.1)
                    # print(t_i)
                    # import ipdb; ipdb.set_trace()
                    break  #Don't try anymore placements
                else:
                    t_i.set_visible(False)




def getTextBbox(text):
    """

    Based on Text.get_window_extent()

    This is called out as a separate function because I expect it will be much more work
    in other plotting toolkits
    """

    visible = text.get_visible()
    text.set_visible(True)
    bbox = text.get_window_extent()
    text.set_visible(visible)
    return bbox


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