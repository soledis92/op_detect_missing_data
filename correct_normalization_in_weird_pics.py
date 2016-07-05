import pdb
import vigra
import numpy as np

#prediction_big_patches = vigra.readHDF5("big_stuff_classifier_512_set_C_prediction_from_LSVM.h5", "data", order = 'C')
#cut_out_big_patches = vigra.readHDF5("big_stuff_classifier_512_set_C_cut_out.h5", "data", order = 'C')
raw = vigra.readHDF5("volume_big_stuff_classifier.h5", "data", order = 'C')
good_pics = vigra.readHDF5("ground_truth_good_full_set.h5", "data", order = 'C')
# processed_pics = raw - cut_out_big_patches
newmin = good_pics[0, :, :].min()
newmax = good_pics[0, :, :].max()
# oldmin = raw[1, :, :].min()
# oldmax = raw[1, :, :].max()
# cutout_min = processed_pics[1, :, :].min()
# cutout_max = processed_pics[1, :, :].max()
# print "new: ", newmin, newmax
# print "old: ", oldmin, oldmax
# print "cutout: ", cutout_min, cutout_max
# adjusted = vigra.colors.contrast(processed_pics.astype(np.float32), 1.0, range=(min, max))
adjusted = vigra.colors.linearRangeMapping(processed_pics.astype(np.float32), oldRange = (oldmin, oldmax), newRange = (newmin, newmax))
vigra.writeHDF5(processed_pics, "processed.h5", "data")
vigra.writeHDF5(adjusted, "adjusted2.h5", "data")
