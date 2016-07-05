import vigra
import numpy

from volumina.pixelpipeline.datasources import LazyflowSource, ArraySource
from volumina.layer import GrayscaleLayer, ColortableLayer, ClickableColortableLayer, AlphaModulatedLayer, RGBALayer

from volumina.api import Viewer
from volumina.colortables import create_random_16bit, create_default_16bit

from PyQt4.QtGui import QApplication, QColor, QKeySequence, QShortcut
import pdb
indir = "./"

datasets = {"raw_bad": "ground_truth_defected_full_set.h5",
            "raw_good": "ground_truth_good_full_set.h5",
            "128_bad": "full_set_defected_prediction_result_128.h5",
            "128_good": "full_set_good_prediction_result_128.h5",
            "150_bad": "full_set_defected_prediction_result_150.h5",
            "150_good": "full_set_good_prediction_result_150.h5",
            "150_v2_bad": "full_set_defected_prediction_result_250_v2.h5",
            "150_chosen_bg": "full_set_defected_prediction_result_250_v3_chosen_bg.h5",
            "150_chosen_bg2": "full_set_defected_prediction_result_250_v3_chosen_bg_v2.h5",
            "150_chosen_bg3": "full_set_defected_prediction_result_250_v3_chosen_bg_v3.h5",
            "cut_out_defected": "full_set_defected_cut_out_result_250_v3_chosen_bg_v3.h5",
            "cut_out_good": "full_set_good_cut_out_result_250_v3_chosen_bg_v3.h5",
            "testing_set_C_prediction": "testing_set_C_prediction_from_LSVM.h5",
            "testing_set_C_raw": "testing_volume_from_C.h5",
            "testing_set_C_cut_out": "testing_set_C_cut_out.h5",
            "c_result": "testing_set_C_object_prediction_result.h5",
            "objects_from_training_defected": "object_prediction_results_from_training_sets_defected.h5",
            "objects_from_training_good": "object_prediction_results_from_training_sets_good.h5",
            "objects_from_testing_set_C": "testing_set_C_cut_out_object_predictions.h5",
            "testing_set_C_segmentation": "segmentation_testing_set_C_pixelwise_also_trained_on_good_filtered.h5",
            "C_segmentation": "testing_set_C_segmentation_filterd_nr1.h5",
            # before correcting borders
            "cremi_test_A_prediction": "cremi_test_A_prediction_from_LSVM.h5",
            #
            "full_img_clasifier_C": "full_img_classifier_segmentation_test_C.h5",
            "check_train_raw_150": "volume_one_wierd_pic.h5",
            "check_train_cutout_150": "check_for_training_150_cutout.h5",
            "check_train_pred_150": "check_for_training_150_prediction_from_LSVM.h5",
            "check_train_raw_200": "volume_one_wierd_pic.h5",
            "check_train_cutout_200": "check_for_training_200_cutout.h5",
            "check_train_pred_200": "check_for_training_200_prediction_from_LSVM.h5",
            "check_train_raw_200.1": "volume_2_wierd_pics.h5",
            "check_train_cutout_200.1": "check_for_training_200.1_cutout.h5",
            "check_train_pred_200.1": "check_for_training_200.1_prediction_from_LSVM.h5",
            "check_train_raw_150.1": "volume_2_wierd_pics.h5",
            "check_train_cutout_150.1": "check_for_training_150.1_cutout.h5",
            "check_train_pred_150.1": "check_for_training_150.1_prediction_from_LSVM.h5",
            "set_C_big_stuff_512": "big_stuff_classifier_512_set_C_prediction_from_LSVM.h5",
            "set_C_150_fixed_borders_pred": "150_fixed_borders_set_C_prediction_from_LSVM.h5",
            "set_C_150_fixed_borders_cutout": "150_fixed_borders_set_C_cutout.h5",
            "segmentation_on_150_biggeroverlap": "segmentation_on_150_biggeroverlap_cutouts.h5",
            "segmentation_big_stuff_classifier": "segmentation_big_stuff_classifier_cutouts.h5",
            "segmentation_on_equalized_image": "segmentation_on_equalized_pic.h5",
            # CREMI train C
            # CREMI test A+
            "cremi_test_A_raw": "cremi_test_A.h5",
            "cremi_test_A_prediction_150_130_30": "cremi_test_A_prediction_150_130_30.h5",
            "cremi_test_A_segmentation": "cremi_test_A_p150_h130_b30_segmentation.h5",
            "cremi_test_A_cutout": "cremi_test_A_cutout_150_130_30.h5",
            "cremi_test_A_objects": "cremi_test_A_objects_.h5",
            "cremi_test_A_objects_png": "cremi_test_A_objects_png.h5",
                ### after tuning:
            "tuning_96_A_predicton": "after_tuning_96_cremi_test_A_prediction_150_30_30.h5",
            "tuning_96_A_cutout": "after_tuning_96_cremi_test_A_cutout_150_30_30.h5",
            "tuning_150_A_predicton": "after_tuning_150_cremi_test_A_prediction_150_30_30.h5",
            # CREMI test B+
            "cremi_test_B_raw": "cremi_test_B.h5",
            "cremi_test_B_pred_512": "cremi_test_B_p512_h350_b30_prediction_from_LSVM.h5",
            "cremi_test_B_cutout_512": "cremi_test_B_p512_h350_b30_cut_out.h5",
            "cremi_test_B_prediction_150_130_30": "cremi_test_B_p150_h130_b30_prediction_from_LSVM.h5",
            "cremi_test_B_segmentation": "cremi_test_B_p150_h130_b30_segmentation.h5",
            "cremi_test_B_cutout": "cremi_test_B_p150_h130_b30_cut_out.h5",
            "cremi_test_B_segmentation_without_bg_correction": "cremi_test_B_p150_h130_b30_segmentation.h5",
            "cremi_test_B_pred_512_C_training": "cremi_test_B_p512_h30_b30_trained_from_C_prediction_from_LSVM.h5",
            "cremi_test_B_cutout_512_C_training": "cremi_test_B_p512_h30_b30_trained_from_C_cut_out.h5",
            "cremi_test_B_pred_512_C_training_equalized": "cremi_test_B_p512_h30_b30_trained_from_C_equalized_prediction_from_LSVM.h5",
            "cremi_test_B_raw_equalized": "cremi_test_B_equalized.h5",
            "cremi_test_B_pred_150_equalized": "pred_150_130_30_equalized.h5",
            "cremi_test_B_big_stuff_increased_bins": "pred_512_130_200.h5",
            "cremi_test_B_objects": "cremi_test_B_objects_.h5",
            "cremi_test_B_objects_png": "cremi_test_B_objects_png.h5",
            # CREMI test C+
            "cremi_test_C_raw": "cremi_test_C_raw.h5",
            "cremi_test_C_prediction_150_130_30": "cremi_test_C_prediction_150_130_30.h5",
            "cremi_test_C_segmentation": "cremi_test_C_p150_h130_b30_segmentation.h5",
            "cremi_test_C_cutout": "cremi_test_C_cutout_150_130_30.h5",
            "cremi_test_C_objects": "cremi_test_C_objects_.h5",
            "cremi_test_C_objects_png": "cremi_test_C_objects_png.h5",
            "cremi_test_C_objects_png_corrected": "cremi_test_C_objects_png_corrected.h5",
            # stuff for bachelor thesis
            "raw": "thesis_defect_types_volume.h5",
            "LSVM_prediction": "thesis_defect_types_LSVM_prediction_150_30_30.h5",
            "cut_out": "thesis_defect_types_cutout_150_30_30.h5",
            "segmentation_pixelwise": "thesis_defect_types_segmentation.h5",
            "segmentation_no_LSVM": "thesis_defect_types_segmentation_full_volume.h5"}


def showStuff(raw_name, pred_viewer1, pred_viewer2, cutout_name, one_extra = None):
    # display the raw and annotations for cremi challenge data
    raw = vigra.impex.readHDF5(indir+datasets[raw_name], "data", order = 'C')
    # raw_old = vigra.readHDF5(indir+datasets["raw_bad"], "data", order = 'C')
    defect_prediction_128 = vigra.impex.readHDF5(indir+datasets[pred_viewer2], "data", order = 'C')
    defect_prediction_150 = vigra.impex.readHDF5(indir+datasets[pred_viewer1], "data", order = 'C')
    cutout_from_150_pred = vigra.impex.readHDF5(indir+datasets[cutout_name], "data", order = 'C')

    ####################################################################################################################
    # only used for fast testing stuff
    #change_one = vigra.readHDF5(indir+datasets["segmentation_on_equalized_image"], "data", order = 'C')
    #pdb.set_trace()
    #defect_prediction_150[1,:,:] = change_one[0,:,:,0]
    ####################################################################################################################
    # defect_prediction_150 = gt[..., 0]
    cutout = numpy.asarray(cutout_from_150_pred)
    rawdata = numpy.asarray(raw)
    # rawdata_old = numpy.asarray(raw_old)
    # op5ify
    # shape5d = rawdata.shape
    shape5d = (1,)+rawdata.shape+(1,)
    print shape5d, rawdata.shape, rawdata.dtype

    app = QApplication([])
    v = Viewer()
    direct = False

    # layer for raw data
    rawdata = numpy.reshape(rawdata, shape5d)
    rawsource = ArraySource(rawdata)
    v.dataShape = shape5d
    lraw = GrayscaleLayer(rawsource, direct=direct)
    lraw.visible = True
    lraw.name = "raw"
    v.layerstack.append(lraw)

    # layer for cutout regions from raw data
    cutout = numpy.reshape(cutout, shape5d)
    cutoutsource = ArraySource(cutout)
    lcutout = GrayscaleLayer(cutoutsource, direct = direct)
    lcutout.visible = False
    lcutout.name = "cut_out"
    v.layerstack.append(lcutout)

    # layer for first prediction result
    defect_prediction_128 = numpy.reshape(defect_prediction_128, shape5d)
    synsource = ArraySource(defect_prediction_128)
    ct = create_random_16bit()
    ct[0] = 0
    lsyn = ColortableLayer(synsource, ct)
    lsyn.name = pred_viewer2
    lsyn.visible = False
    v.layerstack.append(lsyn)

    # layer for second prediction result
    segm = numpy.reshape(defect_prediction_150, shape5d)
    segsource = ArraySource(segm)
    ct = create_random_16bit()
    ct[0] = 0
    lseg = ColortableLayer(segsource, ct)
    lseg.name = pred_viewer1
    lseg.visible = False
    v.layerstack.append(lseg)
    if one_extra is None:
        v.showMaximized()
        app.exec_()

    if one_extra is not None:
        # layer for third prediction result
        extra_prediction = vigra.readHDF5(indir+datasets[one_extra], "data", order = 'C')
        extra_pred_reshaped = numpy.reshape(extra_prediction, shape5d)
        segsource = ArraySource(extra_pred_reshaped)
        ct = create_random_16bit()
        ct[0] = 0
        # ct = create_default_16bit()
        lseg = ColortableLayer(segsource, ct)
        lseg.name = one_extra
        lseg.visible = False
        v.layerstack.append(lseg)
        v.showMaximized()
        app.exec_()


if __name__=="__main__":
    # showStuff("raw_bad", "objects_from_training_defected", "objects_from_training_defected","cut_out_defected")
    # showStuff("raw_good", "objects_from_training_good", "objects_from_training_good", "cut_out_good")


    # check if training with weird pic trains better classifier
    # showStuff("check_train_raw_150", "check_train_pred_150", "check_train_pred_200", "check_train_cutout_150")
    # showStuff("check_train_raw_200", "check_train_pred_200", "check_train_pred_200", "check_train_cutout_200")
    # using 2 weird pics:
    # showStuff("check_train_raw_150.1", "check_train_pred_150.1", "check_train_pred_200.1", "check_train_cutout_150.1")
    # showStuff("check_train_raw_200.1", "check_train_pred_200.1", "check_train_pred_200.1", "check_train_cutout_200.1")

    # testing-set C:
    # without any weird pic in training:
    # showStuff("testing_set_C_raw", "testing_set_C_segmentation", "full_img_clasifier_C", "testing_set_C_cut_out")

    # test on CREMI testing dataset A+ cropped version
    # showStuff("cremi_test_A_raw", "cremi_test_A_prediction", "cremi_test_A_prediction", "cremi_test_A_cutout")

    # test on CREMI testing dataset B+ cropped version
    # LSVM training after histogram equalization
    # showStuff("cremi_test_B_raw", "cremi_test_B_pred_512_C_training", "cremi_test_B_150_segmentation", "cremi_test_B_cutout_512")
    # showStuff("cremi_test_B_raw_equalized", "cremi_test_B_pred_150_equalized", "cremi_test_B_pred_512_C_training_equalized", "cremi_test_B_cutout_512")


########################################################################################################################
#### CREMI testing ###

    # CREMI test A+
    showStuff("cremi_test_A_raw", "cremi_test_A_prediction_150_130_30", "cremi_test_A_segmentation", "cremi_test_A_cutout", "cremi_test_A_objects_png")
    ### after tuning:
    # showStuff("cremi_test_A_raw", "tuning_96_A_predicton", "cremi_test_A_prediction_150_130_30", "tuning_96_A_cutout")
    # CREMI test B+
    # showStuff("cremi_test_B_raw", "cremi_test_B_prediction_150_130_30", "cremi_test_B_segmentation", "cremi_test_B_cutout", "cremi_test_B_objects_png")

    # CREMI test C+
    # showStuff("cremi_test_C_raw", "cremi_test_C_prediction_150_130_30", "cremi_test_C_segmentation", "cremi_test_C_cutout", "cremi_test_C_objects_png")

########################################################################################################################
#### Bachelor thesis stuff ###

    # showStuff("raw", "LSVM_prediction", "segmentation_pixelwise", "cut_out", "segmentation_no_LSVM")
