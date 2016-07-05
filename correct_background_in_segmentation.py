import vigra
import pdb
import numpy as np

# lsvm_mask = vigra.readHDF5("full_set_defected_prediction_result_250_v3_chosen_bg_v3.h5", "data", order = 'C')
lsvm_mask = vigra.readHDF5("thesis_defect_types_LSVM_prediction_150_30_30.h5", "data", order = 'C')
#pdb.set_trace()
segmentation_volume = vigra.readHDF5("thesis_defect_types_segmentation_full_volume.h5", "data", order = 'C')
segmentation_volume = vigra.dropChannelAxis(segmentation_volume)
# sets real background from LSVM-result to 0

for i in reversed(np.unique(segmentation_volume)):
    segmentation_volume[segmentation_volume == i] = i + 1
segmentation_volume[segmentation_volume == 4] = 0
segmentation_volume *= lsvm_mask

vigra.writeHDF5(segmentation_volume, "thesis_defect_types_segmentation_full_volume.h5", "data")
