import vigra
import pdb
import numpy

test = vigra.readHDF5("labels_big_stuff_classifier.h5", "data", order = 'C')
print numpy.unique(test)
for i in [3, 2, 1, 0]:
    test[test == i] = i + 1
print numpy.unique(test)
pdb.set_trace()
vigra.writeHDF5(test, "labels_big_stuff_classifier_no_zero.h5", "data")
