import vigra
import numpy
import pdb

ara = vigra.readHDF5("cremi_test_C_objects_png.h5", "data", order = 'C')
arb = vigra.readHDF5("cremi_test_C_p150_h130_b30_segmentation.h5", "data", order = 'C')

pdb.set_trace()
