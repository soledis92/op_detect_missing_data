import vigra
import numpy
import pdb

object_prediction = vigra.readHDF5("cremi_test_C_objects_png.h5", "data", order = 'C')
for i in reversed(numpy.unique(object_prediction)):
    object_prediction[object_prediction == i] = i + 1
print numpy.unique(object_prediction)
object_prediction[object_prediction == 4] = 0


pdb.set_trace()

vigra.writeHDF5(object_prediction, "cremi_test_C_objects_png_corrected.h5", "data")