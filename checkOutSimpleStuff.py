import pdb
from threading import Lock
from functools import partial
from lazyflow.request import Request, RequestPool
import numpy
import vigra
import h5py
'''
# example of summing up over 5x5 cells in volume
volume = numpy.ones((10,10), dtype = numpy.int8)
cellLength = 5
cells = numpy.zeros((2,2), dtype = numpy.int8)
for y_cell in range(cells.shape[0]):
    for x_cell in range(cells.shape[1]):
        for y in range(y_cell, y_cell+cellLength):
            for x in range(x_cell, x_cell+cellLength):
                cells[y_cell,x_cell] += volume[y,x]

print "vol: "
print volume
print "cells: "
print cells

hist = numpy.histogram(volume, bins = 10)[0]
print hist[0]
array = numpy.ndarray((volume.shape[0], volume.shape[1], 10), dtype = numpy.int8)
print array.shape
print hist
print hist.dtype
for i in range(volume.shape[0]):
    for j in range(volume.shape[1]):
        array[i, j, :] = hist
print array.shape
print array
'''

'''
for i in range(1):
    for j in range(1):
        m = x[i*3:(i+1)*3, j*3:(j+1)*3]
        k = x[i*3+1,j*3+1]
        print m
        print k
'''
'''
val = [1,2,3,4,5,6,6,6,6]
hist = numpy.histogram(val,density = True)[0]
val = [1,2,3,4,5,9,9,9,9]
hist2 = numpy.histogram(val,density = True)[0]
val3 = numpy.ones((10,10))
#print hist
import hickle
d = {}
d[(0,2,0)] = numpy.histogram(val3[0:5,0:5], bins = 15, density = True)[0]
d[(1,2,3)] = numpy.zeros((5,5))
#hickle.dump((enumerate(d.items())), 'test.h5', path = '/volume/test')
r = []
for i,k in enumerate(d.items()):
    r.append(k)
hisFile = 't2.h5'
hickle.dump(r, hisFile, mode = 'a',path = '/volume/test')
hickle.dump(r, hisFile, mode = 'a',path = '/volume/train')
c2 = hickle.load(hisFile, path = '/volume/test')
#c = hickle.load('test.h5')
'''

'''
# calculating the percentage of intersection of 2 sqaures in xy-plane
import math
def percentageOfIntersection(x1, y1, x2, y2, a):
    #  d_x > 0
    if x1 > x2:
        if abs(x2-x1) >= a:
            print "no intersection, dist(x) too large"
            return 0
        # d_y > 0
        if y1 > y2:
            if abs(y2-y1) >= a:
                print "no intersection, dist(y) too large"
                return 0
            print 1
            return math.fabs((x2 - x1 + a) * (y2 - y1 + a)) / (a * a)
        # d_y < 0
        elif y1 < y2:
            if abs(y1-y2) >= a:
                print "no intersection, dist(y) too large"
                return 0
            print 2
            return math.fabs((x2 - x1 + a) * (y2 - y1 - a)) / (a * a)
        # d_y = 0
        else:
            print 3
            return 1 - math.fabs(x1-x2)/a
    # d_x < 0
    elif x1 < x2:
        if abs(x1-x2) >= a:
            print "no intersection"
            return 0
        # d_y > 0
        if y1 > y2:
            if abs(y2-y1) >= a:
                print "no intersection"
                return 0
            print 4
            return math.fabs((x2 - x1 - a) * (y2 - y1 + a)) / (a * a)
        # d_y < 0
        elif y1 < y2:
            if abs(y1-y2) >= a:
                print "no intersection"
                return 0
            print 5
            return math.fabs((x2 - x1 - a) * (y2 - y1 - a)) / (a * a)
        # d_y = 0
        else:
            print 6
            return 1 - math.fabs(x2 - x1) / a
    # d_x = 0
    else:
        # d_y > 0
        if y1 > y2:
            if abs(y2-y1) >= a:
                print "no intersection"
                return 0
            print 7
            return 1 - math.fabs(y2 - y1) / a
        # d_y < 0
        elif y1 < y2:
            if abs(y1-y2) >= a:
                print "no intersection"
                return 0
            print 8
            return 1 - math.fabs(y1 - y2) / a
        # d_y = 0
        else:
            print 9
            return 100

# testing:
assert percentageOfIntersection(2,1,1,0,3) == 4.0/9.0
assert percentageOfIntersection(6,6,0,0,3) == 0
assert percentageOfIntersection(1,0 ,0,2,3) == 2.0/9.0
assert numpy.isclose(percentageOfIntersection(1,0,0,0,7), 42/49.0)
assert percentageOfIntersection(3,4,5,3,3) == 2.0/9.0
assert percentageOfIntersection(3,4,4,5,3) == 4.0/9.0
assert numpy.isclose(percentageOfIntersection(1,0,2,0,3), 6.0/9.0)
assert percentageOfIntersection(0,1,0,0,5) == 20.0/25.0
assert percentageOfIntersection(0,0,0,6,5) == 0
assert percentageOfIntersection(0,0,0,1,5) == 20.0/25.0
assert percentageOfIntersection(3,3,3,3,3) == 100

'''
a = numpy.zeros(5, dtype = numpy.uint8)
b = numpy.ones(5, dtype = numpy.uint8)


