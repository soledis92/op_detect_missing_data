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
    import time
    import logging
    logger = logging.getLogger(__name__)

    t_start = time.time()
    volume = vigra.readHDF5('cremi_test_A.h5', 'data', order = 'C')
    op_detect = OpDetectMissing(graph = Graph())
    op_detect.InputVolume.setValue(volume)
    op_detect.HaloSize.setValue(100)
    op_detect.DetectionMethod.setValue('svm')
    op_detect.NHistogramBins.setValue(30)
    op_detect.positive_TrainingHistograms.setValue(False)
    op_detect.negative_TrainingHistograms.setValue(False)

    op_detect.PatchSize.setValue(150)
    op_detect.OverloadDetector.setValue("2016-06-29_19.58_detector_150_30_30.pkl")
    op_view_test_results = _opDetectMissing(graph = Graph())
    op_view_test_results.input_volume.connect(op_detect.Output)
    result = op_view_test_results.input_volume[:].wait()

    cut_out_volume = result * volume

    vigra.writeHDF5(result, "after_tuning_150_cremi_test_A_prediction_150_30_30.h5", "data")
    vigra.writeHDF5(cut_out_volume, "after_tuning_150_cremi_test_A_cutout_150_30_30.h5", "data")
    '''
    # for viewing results directly
    n = 4
    plt.imshow(result[n, :, :])
    plt.show()
    plt.savefig("patch_" + str(op_detect.PatchSize.value) +
                "_halo_" + str(op_detect.HaloSize.value) +
                "_bins_" + str(op_detect.NHistogramBins.value) +
                "_z" + str(n) +
                "_test_good_volume.png")
    plt.close()
    '''
    t_stop = time.time()

    logger.info("Duration: {}".format(
        time.strftime(
            "%Hh, %Mm, %Ss", time.gmtime((t_stop-t_start) % (24*60*60)))))

    print "done!"
