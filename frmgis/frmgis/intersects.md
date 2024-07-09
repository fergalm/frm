# Implentation details for intersect.py

A common problem encountered in Geographical Information Sytems (GIS) is to determine whether a 2 dimensional point is inside or outside an arbitrary 2d shape. In GDAL, this problem is solved by
[Contains](https://gdal.org/python/osgeo.ogr.Geometry-class.html#Contains) method.

The python interface to GDAL only allows you to call this method on a single point at a time. If you have many points, you can waste a lot of time iterating through the list and incurring the overhead of calling an external library. As a result, calling Contains() in Python can be quite slow.

`intersect.py ` is a fast algorithm for computing which of a set of points is interior to a polygon or a multi-polygon. It relies entirely of fast, vectorised, numpy operations which gives superior performance to other methods.




## Details of the Algorithm.
#### Definitions
Consider a simply connected two dimensional space (i.e one with no holes), and an irregular multi-sided polygon defined in that space (i.e a polygon with >3 lines of possibly unequal length. Further assume that the polygon is **simple**, i.e that is is composed of a single, simple, closed ring, i.e the the first and last points of the ring are the same, that the ring touches itself only at the first and last points, and that the polygon has no holes.

Consider a point at the origin of the space. We wish to decide whether this point is interior to, or exterior to our polygon. We will make use of following theorom

> A line drawn from the point, and stretching to infinity in one direction, will cross the boundaries of the polygon an odd number of times if and only if the point is interior to the polygon.

(not proven here)

This statement is true for any line drawn from the point.

#### Example

	 ______________________________________
	|           _____________             |
	|          |             |            |
	|          |             |            |
	|   o------|-------------|------------|--------
	|__________|             |____________|

**Figure 1:** The point **o** is 	interior the polygon because a line that originates at **o** passes through the edges of the polygon 3 times. In the discussion that follows, **o** will be assumed to be at the origin.

#### First Condition
If the theorem above is true for any line, it is true for horizontal lines pointing in the positive x direction. Horizontal lines are easier to deal with than lines with slope, so we will deal with them exclusively. We will call this horizontal line the **ray**, to distinguish it from all the other lines under discussion.

A Line segment is a portion of a line defined by a start and end point. A simple polygon is defined by a set of connecting line segments, which we will call **edges**. All lines intersect the x-axis, but an edge may only intersect our ray if one y-coordinate has a postive value and the other has a negative value.

In mathematical terms we can write this as
> A line segment can only intersect the ray if (y1 > 0) XOR (y2 > 0)

*a* XOR *b* is only true if one and only one of *a* and *b* are true.

#### Second Condition
While this condition is necessary, it is not sufficient, as Figure 2 shows.


        /
       /
      /
     /   o-------------------
    /
**Figure 2** This diagonal line does not intersect the ray because it crosses the x-axis to the left of the point.

The second condition for intersection is then

> A line segment can only intersect the ray if the crosses the x-axis at a value of x> 0

              |
             /|\
            / | \  b
        a  /  |  \
          /   |   \
    _ ___/____|____\_________
        /     |     \
              |
**Figure 3** Two lines that cross the y-axis at the same intersect. One crosses the x-axis at x>0 and the other at x<0

Examining Figure 3 we can see that a line with positive slope (line *a*) will only intersect the x-axis at x > 0 if and only if the intersect with the y-axis is at y< 0. Similarly, a line with negative slope (line *b*) will only intersect at x>0 iff its intersect is at y<0.  We can write this as

 > A line segment can only intersect the ray if (m > 0) XOR (c > 0)


These two requirements combined are both necessary and sufficient to determine if a given line segment intersects our ray.

We don't know *m* or *c* for our line segments, but we can calculate them. Let us write the equation of the line as

	y  = mx + c			(Eqn 1)




For a line segment defined by the points (x1, y1) and (x2, y2), the slope is defined as

	      m = (y2 - y1)    =  dy
	         -----------     ----
	           (x2-x1)        dx      (x1 != x2)


The problem with this equation is that division is both expensive and unstable in boolean arithmethic. But we don't need to compute the value of *m*, just its sign.

	(m < 0) <=>  (dx > 0) XOR (dy > 0)    (Eqn 2)

For the intersect, let's evaluate its value for the points (x1, y1)

	    y1 = m.x1 + c
	=>  y1 = dy.x1/dx + c
	=>  dx.y1 = dy.x1 + c.dx
	=>  dx.c = dx.y1 - dy.x1

The condition *c* < 0 translates as

	c < 0 <=> (dx.y1 < dyx1) XOR (dx < 0)

Our second condition can now be expressed as

     	(m < 0) XOR (c<0)
	= ((dx > 0) XOR (dy > 0)) XOR (dx.y1 < dy.x1) XOR (dx < 0)

The XOR operator is both comutative and associative

	= ((dx > 0) XOR (dx < 0)) XOR (dy > 0)) XOR ((dx.y1 < dy.x1) XOR (dx < 0))
	= TRUE XOR (dy > 0) XOR (dx.y1 < dyx1)
	= (dy  < 0) XOR (dx.y1 < dyx1)

Assuming y1 != y2.

#### Special cases.
Our equation is only defined for (x1 != x2) and (y1 != y2). In the first case, intersection happens if (x1 > 0).  If y1 = y2, then the line intersects iff y1 == 0.

#### Source code
The condition for intersection can be written as
```python
# & is the AND operator, ^ is the XOR operator
delta = (y1 > 0) ^ (y2 > 0)
alpha = (dy < 0) ^ (dx.y1 < dyx1)
num_crossings = delta & alpha
isInside = (num_crossings % 2 == 1)
```

This algorithm can be neatly expressed in numpy, and can be computed quickly and stably.
For reasons I don't understand, the code works just fine in cases where x1 = x2 xor y1 = y2.
# Generalisation to many points

For a set of points {(x0, y0)}_i, where (x0 != 0) OR (y0 !=0) we simply replace x1, and y1 in the previous equations with x1-x0, and y1-y0, where x1 and x2 are now vectors of points. The code sample above remains unchanged and is effeciently computed with numpy's element-wise operations.


## Performance
In an experiment with 100k points scattered randomly throught the envelope of a roughly square  polygon with 12 sides, Contains computed all points in 2650ms, while the above algorithm needed only 15ms, a speed up of x173.


## Future Improvements
The run time of this algorithm is dominated by memory allocation operations in numpy (60% of the time is spent allocating and assigning new memory). Memory requirements scale as (num target points) x (number of edges in polygon). An algorithm in a compiled language could skip the memory allocations for another ~x2 speedup.
