import vigra
import numpy
import pdb

a = vigra.readHDF5("labels_big_stuff_classifier_also_from_C.h5", "data", order = 'C')
print numpy.unique(a)

print len(a[a==0]) , len(a[a==1]) , len(a[a==2])

print len(a[a==0]) + len(a[a==1]) + len(a[a==2])
print 12*3072*3072

a[a == 3] = 4
a[a == 2] = 3
a[a == 1] = 2
a[a == 0] = 1

print numpy.unique(a)

vigra.writeHDF5(a, "labels_big_stuff_classifier_also_from_C.h5", "data")
