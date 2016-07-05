from lazyflow.graph import Operator, InputSlot, OutputSlot

from lazyflow.graph import Graph

from opDetectMissingData import OpDetectMissing
import HistogramExtractor

import numpy
import vigra
import pdb
from matplotlib import pyplot as plt


class _opDetectMissing(Operator):
    input_volume = InputSlot()

    output = OutputSlot()

    def setupOutputs(self):
        pass

    def propagateDirty(self, slot, subindex, roi):
        # Since this operator doesn't produce data on it's own
        # (all of its output slots are connected directly to other operators),
        # we can rely on 'dirty' events to be propagated from input to output on their own.
        # There's nothing to do here.
        pass

    def execute(self, slot, subindex, roi, result):
        # Since this operator doesn't produce data on it's own
        # (all of its output slots are connected directly to other operators),
        # there's nothing to do here, and this function will never be called.
        assert False, "Shouldn't get here."


if __name__ == "__main__":
    patch_file_combination = {}

    patchsizes = [64, 96, 128, 150, 160, 170, 180, 190, 200, 256]
    filenames = ["2016-05-23_11.01_detector_64_25_30.pkl", "2016-05-23_11.01_detector_96_25_30.pkl", "2016-05-23_11.01_detector_128_25_30.pkl",
                 "2016-05-23_11.01_detector_150_25_30.pkl", "2016-05-23_11.01_detector_160_25_30.pkl", "2016-05-23_11.01_detector_170_25_30.pkl",
                 "2016-05-23_11.01_detector_180_25_30.pkl", "2016-05-23_11.01_detector_190_25_30.pkl", "2016-05-23_11.01_detector_200_25_30.pkl", "2016-05-23_11.01_detector_256_25_30.pkl"]


    patchsize_filename = zip(patchsizes, filenames)
    patchsize_filename_dict = dict(patchsize_filename)
    volume = vigra.readHDF5('test_data_defected_from_ground_truth.h5', 'data').withAxes(*'zyx')
    op_detect = OpDetectMissing(graph = Graph())
    op_detect.InputVolume.setValue(volume)
    op_detect.HaloSize.setValue(25)
    op_detect.DetectionMethod.setValue('svm')
    op_detect.NHistogramBins.setValue(30)
    op_detect.positive_TrainingHistograms.setValue(False)
    op_detect.negative_TrainingHistograms.setValue(False)
    for key in patchsize_filename_dict:
        op_detect.PatchSize.setValue(key)
        op_detect.OverloadDetector.setValue(patchsize_filename_dict[key])
        op_view_test_results = _opDetectMissing(graph = Graph())
        op_view_test_results.input_volume.connect(op_detect.Output)
        result = op_view_test_results.input_volume[:].wait()
        plt.imshow(result[0, :, :])
        plt.savefig(str(key) + "_result.png")
        plt.close()
        print key
    print "done!"
