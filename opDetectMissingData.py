###############################################################################
#   lazyflow: data flow based lazy parallel computation framework
#
#       Copyright (C) 2011-2014, the ilastik developers
#                                <team@ilastik.org>
#
# This program is free software; you can redistribute it and/or
# modify it under the terms of the Lesser GNU General Public License
# as published by the Free Software Foundation; either version 2.1
# of the License, or (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU Lesser General Public License for more details.
#
# See the files LICENSE.lgpl2 and LICENSE.lgpl3 for full text of the
# GNU Lesser General Public License version 2.1 and 3 respectively.
# This information is also available on the ilastik web site at:
# 		  http://ilastik.org/license/
###############################################################################
import logging
from functools import partial
import cPickle as pickle
import tempfile
import re

from lazyflow.graph import Operator, InputSlot, OutputSlot
from lazyflow.stype import Opaque
from lazyflow.request import Request, RequestPool

import numpy as np
import vigra

import HistogramExtractor
import hickle
from sklearn import cross_validation
from skimage import measure

logger = logging.getLogger(__name__)


class VersionError(Exception):
    pass


def extractVersion(s):
    # assuming a string with a decimal number inside
    # e.g. "0.11-ubuntu", "haiku_os-sklearn-0.9"
    reInt = re.compile("\d+")
    m = reInt.findall(s)
    if m is None or len(m)<1:
        raise VersionError("Cannot determine sklearn version")
    else:
        return int(m[1])

try:
    from sklearn import __version__ as sklearn_version
    svcTakesScaleC = extractVersion(sklearn_version) < 11
except (ImportError, VersionError):
    logger.warning("Could not import dependency 'sklearn' for SVMs")
    have_sklearn = False
else:
    have_sklearn = True

def SVC(*args, **kwargs):
    from sklearn.svm import SVC as _SVC
    # old scikit-learn versions take scale_C as a parameter
    # new ones don't and default to True
    if not svcTakesScaleC and "scale_C" in kwargs:
        del kwargs["scale_C"]
    return _SVC(*args, **kwargs)

_defaultBinSize = 30


############################
############################
############################
###                      ###
###  DETECTION OPERATOR  ###
###                      ###
############################
############################
############################


class OpDetectMissing(Operator):
    '''
    Sub-Operator for detection of missing image content
    '''

    InputVolume = InputSlot()
    PatchSize = InputSlot(value=128)
    HaloSize = InputSlot(value=30)
    _overlap_between_two_patches = 0.0
    DetectionMethod = InputSlot(value='classic')
    NHistogramBins = InputSlot(value=_defaultBinSize)
    OverloadDetector = InputSlot(value='')

    # list of training histograms used as positive examples F_p
    positive_TrainingHistograms = InputSlot()
    # list of training histograms used as positive examples F_p
    negative_TrainingHistograms = InputSlot()

    Output = OutputSlot()
    Detector = OutputSlot(stype=Opaque)

    ### PRIVATE class attributes ###
    _manager = None

    ### PRIVATE attributes ###
    _inputRange = (0, 255)
    _needsTraining = False
    _felzenOpts = {"firstSamples": 250, "maxRemovePerStep": 10,
                   "maxAddPerStep": 250, "maxSamples": 1000,
                   "nTrainingSteps": 4}

    def __init__(self, *args, **kwargs):
        super(OpDetectMissing, self).__init__(*args, **kwargs)
        self.positive_TrainingHistograms.setValue(_default_training_histograms())

    def propagateDirty(self, slot, subindex, roi):
        if slot == self.InputVolume:
            self.Output.setDirty(roi)

        if slot == self.positive_TrainingHistograms:
            if not self.positive_TrainingHistograms.value:
                OpDetectMissing._needsTraining = False
            else:
                OpDetectMissing._needsTraining = True

        if slot == self.negative_TrainingHistograms:
            if not self.negative_TrainingHistograms.value:
                OpDetectMissing._needsTraining = False
            else:
                OpDetectMissing._needsTraining = True

        if slot == self.NHistogramBins:
            OpDetectMissing._needsTraining = \
                OpDetectMissing._manager.has(self.NHistogramBins.value)

        if slot == self.PatchSize or slot == self.HaloSize:
            self.Output.setDirty()

        if slot == self.OverloadDetector:
            self.loads(self.OverloadDetector.value)
            self.Output.setDirty()

    def setupOutputs(self):
        self.Output.meta.assignFrom(self.InputVolume.meta)
        self.Output.meta.dtype = np.uint8

        # determine range of input
        if self.InputVolume.meta.dtype == np.uint8:
            r = (0, 255) 
        elif self.InputVolume.meta.dtype == np.uint16:
            r = (0, 65535) 
        else:
            # if no dtype is given, range is set to 0 - maximum of uint64
            r = (0, np.iinfo(np.uint64).max)
        self._inputRange = r

        self.Detector.meta.shape = (1,)

    def execute(self, slot, subindex, roi, result):
        # if detector already exists
        if slot == self.Detector:
            result = self.dumps()
            return result

        # sanity check
        assert self.DetectionMethod.value in ['svm', 'classic'],\
            "Unknown detection method '{}'".format(self.DetectionMethod.value)

        # prefill result
        result_zyxct = vigra.taggedView(
            result, self.InputVolume.meta.axistags).withAxes(*'zyxct')

        # acquire data
        data = self.InputVolume.get(roi).wait()
        data_zyxct = vigra.taggedView(
            data, self.InputVolume.meta.axistags).withAxes(*'zyxct')

        # walk over time and channel axes
        for t in range(data_zyxct.shape[4]):
            for c in range(data_zyxct.shape[3]):
                result_zyxct[..., c, t] = \
                    self._detectMissing(data_zyxct[..., c, t])

        return result

    def _detectMissing(self, data):
        """
        detects missing regions and labels each missing region with 1
        :param data: 3d data with axistags 'zyx'
        :type data: array-like
        """

        # check if input was correct
        assert data.axistags.index('z') == 0 \
            and data.axistags.index('y') == 1 \
            and data.axistags.index('x') == 2 \
            and len(data.shape) == 3, \
            "Data must be 3d with axis 'zyx'."

        _patchSize = self.PatchSize.value
        _haloSize = self.HaloSize.value
        _overlap_between_two_patches = float(_haloSize) / float(_patchSize)
        print "overlap: ", _overlap_between_two_patches

        if _patchSize is None or not _patchSize > 0:
            raise ValueError("__patchSize must be a positive integer")
        if _haloSize is None or _haloSize < 0:
            raise ValueError("__haloSize must be a non-negative integer")

        extractor = HistogramExtractor.HistogramExtractor(
            data, _patchSize, self.NHistogramBins.value, _overlap_between_two_patches)
        histograms_list = extractor.getHistograms()
        histograms = np.vstack(histograms_list)
        # predict via already trained histogram_intersection_kernel-SVM
        # if SVM is untrained the result will be computed using PseudoSVC
        predictions = self.predict(histograms[:, 1], method=self.DetectionMethod.value)

        # write a 1 into every pixel location in the result array, that is corresponding to a defected patch
        result = np.zeros(data.shape, dtype=np.uint8)
        root_locations_of_defected_patches = histograms[np.where(predictions != 0), 0]
        for root_location in root_locations_of_defected_patches[0]:
            xmin = root_location[2] - _patchSize // 2
            xmax = root_location[2] + _patchSize // 2
            ymin = root_location[1] - _patchSize // 2
            ymax = root_location[1] + _patchSize // 2
            # label complete patch region with 1, to indicate defected patch
            result[root_location[0], ymin:ymax, xmin:xmax] = 1
        return result

    def train(self, force=False):
        """
        trains with samples drawn from slot positive_TrainingHistograms and negative_TrainingHistograms
        (retrains only if bin size is currently untrained or force is True)
        :param force: if set True, training is forced no matter if trained detector for current setup already exists.
        """
        # return early if unnecessary
        if not force and not OpDetectMissing._needsTraining and \
                OpDetectMissing._manager.has(self.NHistogramBins.value):
            return

        # return if we don't have any SVMs
        if not have_sklearn:
            return

        logger.debug("Training for {} histogram bins ...".format(
            self.NHistogramBins.value))

        if self.DetectionMethod.value == 'classic' or not have_sklearn:
            # no need to train this
            return

        positive_histograms = self.positive_TrainingHistograms[:].wait()
        negative_histograms = self.negative_TrainingHistograms[:].wait()

        logger.debug("Finished loading histogram data of shape {} as positive examples. And {} as negative examples."
                     "Each histogram has {} bins"
                     .format(positive_histograms.shape, negative_histograms, len(positive_histograms[0][1])))
        assert len(positive_histograms[0][1]) == self.NHistogramBins.value,\
            "Training data has wrong shape (expected: (nSamples,{}), got: {}.".format(
                self.NHistogramBins.value, (positive_histograms.shape[0], len(positive_histograms[0][1])))

        number_of_negative_samples = negative_histograms.shape[0]
        number_of_positive_samples = positive_histograms.shape[0]
        logger.debug(
            "Starting training with " +
            "{} negative patches and {} positive patches...".format(
                number_of_negative_samples, number_of_positive_samples))
        # test for differnet C-values for the svm
        '''
        f = open("c_vs_precision_recall_" + str(patchSize) + "_" + str(haloSize) + "_" + str(binSize) + ".txt", mode = 'a')
        list_of_Cs = []

        for i in range(1, 11, 1):
            list_of_Cs.append(i)
        for i in range(11, 200, 10):
            list_of_Cs.append(i)
        for i in range(100, 1001, 100):
            list_of_Cs.append(i)

        for C_val in list_of_Cs:
            print "C: ", C_val
            precision, error_precision, recall, error_recall = self._cross_validate(positive_histograms, negative_histograms, C_val = C_val)
            print "precision: ", precision, "+-", error_precision
            print "recall: ", recall, "+-", error_recall
            f.write(str(C_val)+'\t'+str(precision)+'\t'+str(error_precision)+'\t'+str(recall)+'\t'+str(error_recall)+'\n')
        f.close()
        '''
        # run with C = 100, since it turned out to be pretty good from patchSize = 64

        precision, error_precision, recall, error_recall =\
            self._cross_validate(positive_histograms, negative_histograms, C_val = 100)
        print "patch-size: ", patchSize
        print "precision: ", precision, "+-", error_precision
        print "recall: ", recall, "+-", error_recall
        f = open("c_vs_precision_recall_" + str(patchSize) + "_" + str(haloSize) + "_" + str(binSize) + ".txt", mode = 'a')
        f.write(str(binSize) + '\t' + str(precision) + '\t' + str(error_precision) + '\t' + str(recall) + '\t' + str(error_recall) + '\n')

        self._felzenszwalbTraining(negative_histograms[:, 1], positive_histograms[:, 1], C_val = 100)
        logger.debug("Finished training.")
        OpDetectMissing._needsTraining = False

    def _felzenszwalbTraining(self, negative, positive, C_val = 100):
        """
        we want to train on a 'hard' subset of the training data, see
        FELZENSZWALB ET AL.: OBJECT DETECTION WITH DISCRIMINATIVELY TRAINED PART-BASED MODELS (4.4), PAMI 32/9
        data-mining hard examples -> more efficient training
        """

        # TODO sanity checks
        method = self.DetectionMethod.value

        # set options for Felzenszwalb training
        firstSamples = self._felzenOpts["firstSamples"]
        maxRemovePerStep = self._felzenOpts["maxRemovePerStep"]
        maxAddPerStep = self._felzenOpts["maxAddPerStep"]
        maxSamples = self._felzenOpts["maxSamples"]
        nTrainingSteps = self._felzenOpts["nTrainingSteps"]

        # initial choice of training samples
        (initNegative, choiceNegative, _, _) = \
            _choose_random_subset(negative, min(firstSamples, len(negative)))
        (initPositive, choicePositive, _, _) = \
            _choose_random_subset(positive, min(firstSamples, len(positive)))
        # setup for parallel training
        samples = [negative, positive]
        choice = [choiceNegative, choicePositive]
        S_t = [initNegative, initPositive]

        finished = [False, False]

        ### BEGIN SUBROUTINE ###
        def felzenstep(x, cache, ind):

            case = ("positive" if ind > 0 else "negative") + " set"
            pred = self.predict(x, method=method)

            hard = np.where(pred != ind)[0]
            easy = np.setdiff1d(range(len(x)), hard)
            logger.debug(" {}: currently {} hard and {} easy samples".format(
                case, len(hard), len(easy)))

            # shrink the cache
            easy_in_cache = np.intersect1d(easy, cache) if len(easy) > 0 else []
            if len(easy_in_cache) > 0:
                (remove_from_cache, _, _, _) = _choose_random_subset(
                    easy_in_cache, min(len(easy_in_cache), maxRemovePerStep))
                cache = np.setdiff1d(cache, remove_from_cache)
                logger.debug(" {}: shrunk the cache by {} elements".format(
                    case, len(remove_from_cache)))

            # grow the cache
            temp = len(cache)
            add_to_cache = _choose_random_subset(
                hard, min(len(hard), maxAddPerStep))[0]
            cache = np.union1d(cache, add_to_cache)
            added_hard = len(cache)-temp
            logger.debug(" {}: grown the cache by {} elements".format(
                case, added_hard))

            if len(cache) > maxSamples:
                logger.debug(
                    " {}: Cache to big, removing elements.".format(case))
                cache = _choose_random_subset(cache, maxSamples)[0]

            # apply the cache
            C = x[cache]

            return C, cache, added_hard == 0
        ### END SUBROUTINE ###

        ### BEGIN PARALLELIZATION FUNCTION ###
        def partFun(i):
            (C, new_choice, new_finished) = felzenstep(samples[i], choice[i], i)
            S_t[i] = C
            choice[i] = new_choice
            finished[i] = new_finished
        ### END PARALLELIZATION FUNCTION ###

        for k in range(nTrainingSteps):

            logger.debug(
                "Felzenszwalb Training " +
                "(step {}/{}): {} hard negative samples, {}".format(
                    k+1, nTrainingSteps, len(S_t[0]), len(S_t[1])) +
                "hard positive samples.")
            # update the SVM model
            self.fit(S_t[0], S_t[1], method=method, C_val = C_val)

            pool = RequestPool()
            # data mine hard examples (felzenszwalb) in parallel
            for i in range(len(S_t)):
                req = Request(partial(partFun, i))
                pool.add(req)

            pool.wait()
            pool.clean()

            if np.all(finished):
                # already have all hard examples in training set
                break
        # train the SVM with the final choice of positive + negative samples
        self.fit(S_t[0], S_t[1], method=method, C_val = C_val)

        logger.debug(" Finished Felzenszwalb Training.")

    def _cross_validate(self, positive_histograms, negative_histograms, n_folds = 10, C_val = 100):
        """
        N-fold cross validation based on splitting by sklearn's cross_validation.KFold() object.
        Training and testing is done with a HIK-LSVM.

        :param positive_histograms:
        :param negative_histograms:
        :param n_folds:
        :param C_val:
        :return: precision +- error, recall +- error
        """

        # prepare n-fold cross validation
        logger.info("{}-fold cross-validation".format(n_folds))
        list_of_precisions_from_validation_cycles_pixelwise = []
        list_of_recalls_from_validation_cycles_pixelwise = []
        list_of_precisions_from_validation_cycles_objectwise = []
        list_of_recalls_from_validation_cycles_objectwise = []

        # split up into groups for training and testing
        positive_folds = cross_validation.KFold(positive_histograms.shape[0], n_folds, shuffle = True)
        negative_folds = cross_validation.KFold(negative_histograms.shape[0], n_folds, shuffle = True)
        # a bit ugly way of making sure, that we take positive and negative training samples for the training
        # necessary because the split up groups generated via sklearns KFold cant be accessed via indexing.
        # Similar way to enumerating the training sets
        pos_counter = 0
        neg_counter = 0

        for positive_trainig_group_indices, positive_testing_group_indices in positive_folds:
            for negative_trainig_group_indices, negative_testing_group_indices in negative_folds:
                # same set number for positive and negative samples
                if pos_counter == neg_counter:
                    # do the training
                    self._felzenszwalbTraining(negative_histograms[negative_trainig_group_indices, 1],
                                               positive_histograms[positive_trainig_group_indices, 1], C_val)
                    # do the testing
                    # classify the testing sub-set
                    predictions_for_negative_sub_set =\
                        self.predict(negative_histograms[negative_testing_group_indices, 1],
                                     method = self.DetectionMethod.value)
                    predictions_for_positive_sub_set =\
                        self.predict(positive_histograms[positive_testing_group_indices, 1],
                                     method = self.DetectionMethod.value)
                    # get all positions of interest
                    predicted_detections_from_positive_sub_group =\
                        positive_histograms[np.where(predictions_for_positive_sub_set == 1), 0]
                    predicted_background_from_positive_sub_group =\
                        positive_histograms[np.where(predictions_for_positive_sub_set == 0), 0]
                    predicted_detections_from_negative_sub_group =\
                        negative_histograms[np.where(predictions_for_negative_sub_set == 0), 0]
                    predicted_background_from_negative_sub_group =\
                        negative_histograms[np.where(predictions_for_negative_sub_set == 1), 0]

                    # count how many are wrong pixelwise
                    wrongly_predicted_detections_positive_sub_group_pixelwise =\
                        self._count_wrong_pixelwise(predicted_detections_from_positive_sub_group, 1)
                    wrongly_predicted_background_positive_sub_group_pixelwise =\
                        self._count_wrong_pixelwise(predicted_background_from_positive_sub_group, 0)
                    wrongly_predicted_detections_negative_sub_group_pixelwise =\
                        self._count_wrong_pixelwise(predicted_detections_from_negative_sub_group, 0)
                    wrongly_predicted_background_negative_sub_group_pixelwise =\
                        self._count_wrong_pixelwise(predicted_background_from_negative_sub_group, 1)

                    # count how many are wrong objectwise
                    wrongly_predicted_detections_positive_sub_group_objectwise =\
                        self._count_wrong_objectwise(predicted_detections_from_positive_sub_group, 1)
                    wrongly_predicted_background_positive_sub_group_objectwise =\
                        self._count_wrong_objectwise(predicted_background_from_positive_sub_group, 0)
                    wrongly_predicted_detections_negative_sub_group_objectwise =\
                        self._count_wrong_objectwise(predicted_detections_from_negative_sub_group, 0)
                    wrongly_predicted_background_negative_sub_group_objectwise =\
                        self._count_wrong_objectwise(predicted_background_from_negative_sub_group, 1)

                    # calculate precision, recall for pixelwise measurement
                    true_positives_pixelwise = len(predicted_detections_from_positive_sub_group[0]) +\
                        len(predicted_detections_from_negative_sub_group[0]) -\
                        wrongly_predicted_detections_positive_sub_group_pixelwise -\
                        wrongly_predicted_detections_negative_sub_group_pixelwise
                    false_positives_pixelwise = wrongly_predicted_detections_positive_sub_group_pixelwise +\
                        wrongly_predicted_detections_negative_sub_group_pixelwise
                    false_negatives_pixelwise = wrongly_predicted_background_positive_sub_group_pixelwise +\
                        wrongly_predicted_background_negative_sub_group_pixelwise

                    precision_pixelwise = float(true_positives_pixelwise) /\
                        float(true_positives_pixelwise + false_positives_pixelwise)
                    recall_pixelwise = float(true_positives_pixelwise) /\
                        float(true_positives_pixelwise + false_negatives_pixelwise)

                    # calculate precision, recall for objectwise measurement
                    true_positives_objectwise = len(predicted_detections_from_positive_sub_group[0]) +\
                        len(predicted_detections_from_negative_sub_group[0]) -\
                        wrongly_predicted_detections_positive_sub_group_objectwise -\
                        wrongly_predicted_detections_negative_sub_group_objectwise
                    false_positives_objectwise = wrongly_predicted_detections_positive_sub_group_objectwise +\
                        wrongly_predicted_detections_negative_sub_group_objectwise
                    false_negatives_objectwise = wrongly_predicted_background_positive_sub_group_objectwise +\
                        wrongly_predicted_background_negative_sub_group_objectwise

                    precision_objectwise = float(true_positives_objectwise) /\
                        float(true_positives_objectwise + false_positives_objectwise)
                    recall_objectwise = float(true_positives_objectwise) /\
                        float(true_positives_objectwise + false_negatives_objectwise)

                    # append current precision, recall to corresponding lists
                    list_of_precisions_from_validation_cycles_pixelwise.append(precision_pixelwise)
                    list_of_precisions_from_validation_cycles_objectwise.append(precision_objectwise)
                    list_of_recalls_from_validation_cycles_pixelwise.append(recall_pixelwise)
                    list_of_recalls_from_validation_cycles_objectwise.append(recall_objectwise)
                    neg_counter += 1
                else:
                    neg_counter += 1
            neg_counter = 0
            pos_counter += 1

        # calculate averages for precision and recall values
        array_of_precisions_pixelwise = np.asarray(list_of_precisions_from_validation_cycles_pixelwise)
        array_of_precisions_objectwise = np.asarray(list_of_precisions_from_validation_cycles_objectwise)
        array_of_recalls_pixelwise = np.asarray(list_of_recalls_from_validation_cycles_pixelwise)
        array_of_recalls_objectwise = np.asarray(list_of_recalls_from_validation_cycles_objectwise)

        mean_precision_pixelwise = array_of_precisions_pixelwise.mean()
        mean_precision_objectwise = array_of_precisions_objectwise.mean()
        mean_recall_pixelwise = array_of_recalls_pixelwise.mean()
        mean_recall_objectwise = array_of_recalls_objectwise.mean()

        error_precision_pixelwise = array_of_precisions_pixelwise.std()
        error_precision_objectwise = array_of_precisions_objectwise.std()
        error_recall_pixelwise = array_of_recalls_pixelwise.std()
        error_recall_objectwise = array_of_recalls_objectwise.std()
        return mean_precision_objectwise, error_precision_objectwise, mean_recall_objectwise, error_recall_objectwise

    #####################
    ### CLASS METHODS ###
    #####################

    @classmethod
    def fit(cls, negative, positive, method='classic', C_val= 100):
        """
        train the underlying SVM
        :param C_val: c-parameter for training the svm
        :param negative: histograms for negative samples
        :param positive: histograms for positive samples
        :param method: clssification method (SVM, 'classic' if no SVM available)
        """

        if cls._manager is None:
            cls._manager = SVMManager()

        if method == 'classic' or not have_sklearn:
            return
        assert len(negative[0]) == len(positive[0]), \
            "Negative and positive histograms must have the same number of bins."

        n_bins = len(negative[0])

        # relabel step. where all positive samples are labeled 1, all negative samples are labeled 0
        labels = np.zeros(len(negative)+len(positive), dtype = np.int8)
        labels[len(negative):] = 1
        # samples needed in shape: (nSamples, nFeatures), whereas nFeatures is #ofBins in our case
        samples = np.hstack((negative, positive))

        samples_reshaped_for_svm = np.zeros((len(samples), len(samples[0])))
        for i in range(len(samples)):
            samples_reshaped_for_svm[i, :] = samples[i]

        svm = SVC(C=C_val, kernel=_histogramIntersectionKernel, scale_C=True)
        svm.fit(samples_reshaped_for_svm, labels)
        cls._manager.add(svm, n_bins, overwrite=True)

    @classmethod
    def predict(cls, X, method='classic'):
        """
        predict if the histograms in X correspond to missing regions
        do this for subsets of X in parallel
        :param method, either 'classic' or 'svm'. If 'classic', PseudoSVC is used, otherwise SVM with HIK is used.
        :return numpy array
        """
        if cls._manager is None:
            cls._manager = SVMManager()

        # svm input has to be (nSamples, nFeatures) -> for us: (nSampels = len(X), nFeatures = # of histogrambins )
        X_reshaped = np.zeros((len(X), len(X[0])))
        for i in range(len(X)):
            X_reshaped[i, :] = X[i]
        n_bins = len(X[0])

        if method == 'classic' or not have_sklearn:
            logger.warning("no real svm used! -> PseudoSVC")
            svm = PseudoSVC()
        else:
            # load samples for histograms of labeled regions
            try:
                svm = cls._manager.get(n_bins)
            except SVMManager.NotTrainedError:
                # fail gracefully if not trained => responsibility of user!
                svm = PseudoSVC()
        y = np.zeros((len(X),)) * np.nan

        pool = RequestPool()

        # chunk up all samples from X into chunks that will be predicted in parallel
        chunk_size = 1000  # FIXME magic number??
        n_chunks = len(X)/chunk_size + (1 if len(X) % chunk_size > 0 else 0)

        s = [slice(k * chunk_size, min((k + 1) * chunk_size, len(X)))
             for k in range(n_chunks)]

        def partial_function(i):
            y[s[i]] = svm.predict(X_reshaped[s[i], :])

        for i in range(n_chunks):
            req = Request(partial(partial_function, i))
            pool.add(req)

        pool.wait()
        pool.clean()
        return np.asarray(y)

    @classmethod
    def _count_wrong_pixelwise(cls, positions_matrix, bool_label):
        # set wanted labels to either background or foreground labels
        if not bool_label:
            labels_to_check = [1]
        else:
            labels_to_check = [2, 3]

        wrong_classifications = 0
        # walk over patches corresponding to current position
        for position in positions_matrix[0]:
            xmin = position[2] - patchSize/2
            xmax = position[2] + patchSize/2
            ymin = position[1] - patchSize/2
            ymax = position[1] + patchSize/2
            for label in labels_to_check:
                if not bool_label:
                    # looking for background, so no labeled pixels should occour
                    if not set(np.unique(labels[position[0], ymin:ymax, xmin:xmax])).issubset(set(labels_to_check)):
                        wrong_classifications += 1
                else:
                    # looking for foreground labels, so at least one labeled pixel should occour in patch
                    if label not in np.unique(labels[position[0], ymin:ymax, xmin:xmax]):
                        wrong_classifications += 1
        return wrong_classifications

    @classmethod
    def _count_wrong_objectwise(cls, positions_matrix, bool_label, minimum_component_size = 30):
        foreground_labels = [2, 3]
        wrong_classifications = 0
        # check each patch corresponding to a root-location
        for position in positions_matrix[0]:
            found_at_least_one = False
            xmin = position[2] - patchSize/2
            xmax = position[2] + patchSize/2
            ymin = position[1] - patchSize/2
            ymax = position[1] + patchSize/2

            # compute connected components using 8 neighborhood-cc- labeleing
            cc = measure.label(labels[position[0], ymin:ymax, xmin:xmax], connectivity = 2, background = 1)
            if bool_label:
                for component in set(np.unique(cc)).difference({-1, 0}):
                    if len(labels[position[0], ymin:ymax, xmin:xmax][np.where(cc == component)]) > minimum_component_size\
                            and set(set(labels[position[0], ymin:ymax, xmin:xmax][np.where(cc == component)])).issubset(set(foreground_labels)):
                        # found a real foreground-labeled-detection
                        found_at_least_one = True
            # if looking for foreground
            if bool_label:
                if not found_at_least_one:
                    wrong_classifications += 1
            # if looking for background
            else:
                if found_at_least_one:
                    wrong_classifications += 1

        return wrong_classifications

    @classmethod
    def has(cls, n, method='classic'):

        if cls._manager is None:
            cls._manager = SVMManager()
        logger.debug(str(cls._manager))

        if method == 'classic' or not have_sklearn:
            return True
        return cls._manager.has(n)

    @classmethod
    def reset(cls):
        cls._manager = SVMManager()
        logger.debug("Reset all detectors.")

    @classmethod
    def dumps(cls):

        if cls._manager is None:
            cls._manager = SVMManager()
        return pickle.dumps(cls._manager.extract())

    @classmethod
    def loads(cls, detector_file_name):

        if cls._manager is None:
            cls._manager = SVMManager()

        if len(detector_file_name) > 0:
            detector_file = open(detector_file_name)
            try:
                unpickeled_detector = pickle.load(detector_file)
            except Exception as err:
                logger.error(
                    "Failed overloading detector due to an error: {}".format(
                        str(err)))
                return
            cls._manager.overload(unpickeled_detector)
            logger.debug("Loaded detector: {}".format(str(cls._manager)))


#############################
#############################
#############################
###                       ###
###         TOOLS         ###
###                       ###
#############################
#############################
#############################


class PseudoSVC(object):
    # pseudo SVM, used in case there is no real SVM
    def __init__(self, *args, **kwargs):
        pass

    def fit(self, *args, **kwargs):
        pass

    @staticmethod
    def predict(self, *args, **kwargs):
        logger.warning("prediction was done with PseudoSVC!!!")
        x = args[0]
        out = np.zeros(len(x))
        for k, patch in enumerate(x):
            out[k] = 1 if np.all(patch[patch[1:] == 0]) else 0
        return out


class SVMManager(object):
    """
    manages our SVMs for multiple bin numbers
    used for:
        * checking if a svm for a certain bin number is trained
        * adding a svm (specific bin number) to the current detector
        * loading a svm from a certain detector
        * extracting a detector from a file
        * removing a certain svm from a detector
    """

    _svms = None

    class NotTrainedError(Exception):
        pass

    def __init__(self):
        self._svms = {'version': 1}

    def get(self, n):
        try:
            return self._svms[n]
        except KeyError:
            raise self.NotTrainedError(
                "Detector for bin size {} not trained.\nHave {}.".format(
                    n, self._svms))

    def add(self, svm, n, overwrite=False):
        if n not in self._svms.keys() or overwrite:
            self._svms[n] = svm

    def remove(self, n):
        try:
            del self._svms[n]
        except KeyError:
            # don't fail, just complain
            logger.error("Tried removing a detector which is not trained yet.")

    def has(self, n):
        return n in self._svms

    def extract(self):
        return self._svms

    def overload(self, obj):
        if 'version' in obj and obj['version'] == self._svms['version']:
            self._svms = obj
            return
        else:
            try:
                for n in obj['svm'].keys():
                    for svm in obj['svm'][n].values():
                        self.add(svm, n, overwrite=True)
            except KeyError:
                # don't fail, just complain
                logger.error(
                    "Detector overload format not recognized, "
                    "no detector loaded.")

    def __str__(self):
        return str(self._svms)


def _choose_random_subset(data, n):
    choice = np.random.permutation(len(data))
    return data[choice[:n]], choice[:n], data[choice[n:]], choice[n:]


def _patchify(data, patch_size, halo_size):
    """
    grids the input data into patches and slices
    data must be 2D y-x
    returns (patches, slices)
    """

    patches = []
    slices = []
    n_patches_x = data.shape[1] / patch_size + (1 if data.shape[1] % patch_size > 0 else 0)
    n_patches_y = data.shape[0] / patch_size + (1 if data.shape[0] % patch_size > 0 else 0)

    for y in range(n_patches_y):
        for x in range(n_patches_x):
            right = min((x+1) * patch_size + halo_size, data.shape[1])
            bottom = min((y+1) * patch_size + halo_size, data.shape[0])

            right_is_incomplete = (x+1) * patch_size > data.shape[1]
            bottom_is_incomplete = (y+1) * patch_size > data.shape[0]

            left = max(x * patch_size - halo_size, 0) if not right_is_incomplete \
                else max(0, right - patch_size - halo_size)
            top = max(y * patch_size - halo_size, 0) if not bottom_is_incomplete \
                else max(0, bottom - patch_size - halo_size)

            patches.append(data[top:bottom, left:right])

            if right_is_incomplete:
                horizontal_slice = slice(
                    max(data.shape[1] - patch_size, 0), data.shape[1])
            else:
                horizontal_slice = slice(patch_size * x, patch_size * (x + 1))

            if bottom_is_incomplete:
                vertical_slice = slice(
                    max(data.shape[0] - patch_size, 0), data.shape[0])
            else:
                vertical_slice = slice(patch_size * y, patch_size * (y + 1))

            slices.append((vertical_slice, horizontal_slice))
    return patches, slices


def _histogramIntersectionKernel(X, Y):
    """
    implements the histogram intersection kernel in a fancy way
    (standard: k(x,y) = sum(min(x_i,y_i)) )

    Restriction from sklearn:
    Input must be two arrays shaped as: (nSamples_X, nFeatures_X, nSamples_y, nFeatures_Y)
    Output must be kernel-matrix of shape (nSamples_X, nSamples_Y)
    """

    A = X.reshape((X.shape[0], 1, X.shape[1]))
    B = Y.reshape((1, ) + Y.shape)

    return np.sum(np.minimum(A, B), axis=2)


def _default_training_histograms():
    """
    produce a standard training set with black regions
    """

    n_hists = 100
    n = _defaultBinSize+1
    hists = np.zeros((n_hists, n))

    # generate n_hists/2 positive sets
    for i in range(n_hists/2):
        (hists[i, :n-1], _) = np.histogram(
            np.zeros((64, 64), dtype=np.uint8), bins=_defaultBinSize,
            range=(0, 255), density=True)
        hists[i, n-1] = 1

    for i in range(n_hists/2, n_hists):
        (hists[i, :n-1], _) = np.histogram(
            np.random.random_integers(60, 180, (64, 64)), bins=_defaultBinSize,
            range=(0, 255), density=True)

    return hists

############################
############################
############################
###                      ###
###         MAIN         ###
###                      ###
############################
############################
############################


if __name__ == "__main__":

    import argparse
    import os.path
    from sys import exit
    import time

    from lazyflow.graph import Graph

    from lazyflow.operators.opDetectMissingData import _histogramIntersectionKernel

    logging.basicConfig()
    logger.setLevel(logging.INFO)

    thisTime = time.strftime("%Y-%m-%d_%H.%M")
    # BEGIN ARGPARSE

    parser = argparse.ArgumentParser(
        description='Train a missing slice detector'+
        """
        Example invocation:
        python2 opDetectMissingData.py block1_test.h5 block1_testLabels.h5 --patch 64 --halo 32 --bins 30 -d ~/testing/2013_08_16 -t 9-12 --opts 200,0,400,1000,2 --shape "(1024,1024,14)"
        """)

    parser.add_argument(
        'file', nargs='*', action='store',
        help="volume and labels (if omitted, the working directory must contain histogram files)")

    parser.add_argument(
        '--testvolume', dest = 'testvolume', action = 'store', default = None,
        help='testing volume must be 3d data that is only used for prediction purpose')

    parser.add_argument(
        '-d', '--directory', dest='directory', action='store', default="/tmp",
        help='working directory, histograms and detector file will be stored there')

    parser.add_argument(
        '-t', '--testingrange', dest='testingrange', action='store', default=None,
        help='the z range of the labels that are for testing'
             '(like "0-3,11,17-19" which would evaluate to [0,1,2,3,11,17,18,19])')

    parser.add_argument(
        '-f', '--force', dest='force', action='store_true', default=False,
        help='force extraction of histograms, even if the directory already contains histograms')

    parser.add_argument(
        '--patch', dest='patchSize', action='store', default='64',
        help='patch size (e.g.: "32,64-128")')
    parser.add_argument(
        '--halo', dest='haloSize', action='store', default='64',
        help='halo size (e.g.: "32,64-128")')
    parser.add_argument(
        '--bins', dest='binSize', action='store', default='30',
        help='number of histogram bins (e.g.: "10-15,20")')

    parser.add_argument(
        '--shape', dest='shape', action='store', default=None, 
        help='shape of the volume in tuple notation "(x,y,z)" (only neccessary if loading histograms from file)')

    parser.add_argument(
        '--opts', dest='opts', action='store', default='250,0,250,1000,4',
        help='<initial number of samples>,<maximum number of samples removed per step>,'
             '<maximum number of samples added per step>,' +
             '<maximum number of samples>,<number of steps> (e.g. 250,0,250,1000,4)')

    args = parser.parse_args()

    # END ARGPARSE

    # BEGIN FILESYSTEM

    working_directory = args.directory
    assert os.path.isdir(working_directory), \
        "Directory '{}' does not exist.".format(working_directory)
    for f in args.file:
        assert os.path.isfile(f), "'{}' does not exist.".format(f)

    # END FILESYSTEM

    # BEGIN NORMALIZE

    def _expand(range_list):
        if range_list is not None:
            single_ranges = range_list.split(',')
            expanded_ranges = []
            for r in single_ranges:
                r2 = r.split('-')
                if len(r2) == 1:
                    expanded_ranges.append(int(r))
                elif len(r2) == 2:
                    for i in range(int(r2[0]), int(r2[1])+1):
                        expanded_ranges.append(i)
                else:
                    logger.error("Syntax Error: '{}'".format(r))
                    exit(33)
            return np.asarray(expanded_ranges)
        else:
            return np.zeros((0,))

    test_range = _expand(args.testingrange)

    patchSizes = _expand(args.patchSize)
    haloSizes = _expand(args.haloSize)
    binSizes = _expand(args.binSize)

    try:
        opts = [int(opt) for opt in args.opts.split(",")]
        assert len(opts) == 5
        opts = dict(zip(
            ["firstSamples", "maxRemovePerStep", "maxAddPerStep",
             "maxSamples", "nTrainingSteps"], opts))
    except:
        raise ValueError(
            "Cannot parse '--opts' argument '{}'".format(args.opts))

    # END NORMALIZE

    # instantiate detection-operator
    op = OpDetectMissing(graph=Graph())
    op._felzenOpts = opts

    logger.info("Starting training script ({})".format(
        time.strftime("%Y-%m-%d %H:%M")))
    t_start = time.time()

    # iterate training conditions
    for patchSize in patchSizes:
        for haloSize in haloSizes:
            for binSize in binSizes:

                histfile = os.path.join(
                    working_directory,
                    "histograms_%d_%d_%d.h5" % (patchSize, haloSize, binSize))
                detfile = os.path.join(
                    working_directory,
                    "%s_detector_%d_%d_%d.pkl" % (
                        thisTime, patchSize, haloSize, binSize))
                predfile = os.path.join(
                    working_directory,
                    "%s_prediction_results_%d_%d_%d.h5" % (
                        thisTime, patchSize, haloSize, binSize))

                startFromLabels = args.force or not os.path.exists(histfile)

                # EXTRACT HISTOGRAMS
                if startFromLabels:
                    logger.info("Gathering histograms from {} patches (this could take a while) ...".format(
                        (patchSize, haloSize, binSize)))
                    assert len(args.file) == 2, \
                        "If there are no histograms available, volume and labels must be provided."

                    locations = ['/volume/data', '/data', '/cube']

                    volume = None
                    labels = None

                    for location in locations:
                        try:
                            volume = vigra.impex.readHDF5(args.file[0], location).withAxes(*'zyx')
                            break
                        except KeyError:
                            pass
                    if volume is None:
                        logger.error(
                            "Could not find a volume in {} with paths {}".format(
                                args.file[0], locations))
                        exit(42)

                    for location in locations:
                        try:
                            labels = vigra.impex.readHDF5(
                                args.file[1], location).withAxes(*'zyx')
                            break
                        except KeyError:
                            pass
                    if labels is None:
                        logger.error(
                            "Could not find a volume in {} with paths {}".format(
                                args.file[1], locations))
                        exit(43)

                    volShape = volume.withAxes(*'xyz').shape

                    # bear with me, complicated axistags stuff is neccessary
                    # for my old vigra to work
                    trainrange = np.setdiff1d(
                        np.arange(volume.shape[0]), test_range)

                    trainData = vigra.taggedView(
                        volume[trainrange, :, :],
                        force = True,
                        axistags = volume.axistags)
                    trainLabels = vigra.taggedView(
                        labels[trainrange, :, :],
                        force = True,
                        axistags = labels.axistags)

                    # extract histograms from volume at as 2 | 3 labeled pixels as root-locations -> positive examples
                    # FIxme support range of labels that count as a positive detection
                    trainHistogramExtractor_l2 = HistogramExtractor.HistogramExtractorFromLabels(
                        trainData, trainLabels, 2, patchSize, binSize, 100000)
                    trainHistogramExtractor_l3 = HistogramExtractor.HistogramExtractorFromLabels(
                        trainData, trainLabels, 3, patchSize, binSize, 100000)
                    # extract histograms from volume at as 1 labeled pixels as root-locations -> negative examples
                    trainHistogramExtractor_l1 = HistogramExtractor.HistogramExtractorFromLabels(
                    trainData, trainLabels, 1, patchSize, binSize, 5000000)
                    trainHistogramExtractor_l4 = HistogramExtractor.HistogramExtractorFromLabels(
                        trainData, trainLabels, 4, patchSize, binSize, 10000)

                    positive_train_histograms2 = trainHistogramExtractor_l2.getHistograms()
                    positive_train_histograms3 = trainHistogramExtractor_l3.getHistograms()
                    positive_train_histograms = positive_train_histograms2 + positive_train_histograms3
                    negative_train_histograms4 = trainHistogramExtractor_l4.getHistograms()
                    negative_train_histograms1 = trainHistogramExtractor_l1.getHistograms()
                    negative_train_histograms = negative_train_histograms1 + negative_train_histograms4
                    # convert to list, to have a by "hickle" supported type for hdf5 storage
                    # using hickle, because vigra.writeHDF5 doesn't support any list-like type
                    hickle.dump(positive_train_histograms, histfile, mode = 'a', path = '/volume/train/positives')
                    hickle.dump(negative_train_histograms, histfile, mode = 'a', path = '/volume/train/negatives')

                    logger.info("Dumped histograms to '{}'.".format(histfile))
                else:
                    logger.info("Gathering histograms from file...")
                    positive_train_histograms = hickle.load(histfile, path = '/volume/train/positives')
                    negative_train_histograms = hickle.load(histfile, path = '/volume/train/negatives')
                    logger.info("Loaded histograms from '{}'.".format(
                        histfile))

                    # check if all values in each histogram are valid (!= nan | inf)
                    for sample in positive_train_histograms:
                        assert not np.any(np.isinf(sample[1]))
                        assert not np.any(np.isnan(sample[1]))
                    for sample in negative_train_histograms:
                        assert not np.any(np.isinf(sample[1]))
                        assert not np.any(np.isnan(sample[1]))
                '''
                ########################################################################################################
                # uncomment this only for development of cross validation!!!
                # labels are needed to check if true/ false positive, but the should only be accessable in training
                labels = None
                locations = ['/volume/data', '/cube', '/data']
                for location in locations:
                        try:
                            labels = vigra.impex.readHDF5(
                                args.file[1], location).withAxes(*'zyx')
                            break
                        except KeyError:
                            pass
                if labels is None:
                    logger.error(
                        "Could not find a volume in {} with paths {}".format(args.file[1], locations))
                ########################################################################################################
                '''
                # TRAIN
                logger.info("Training...")
                # set up cell size
                op.PatchSize.setValue(patchSize)
                op.HaloSize.setValue(haloSize)
                # set up resolution of the response per root-filter (response = histogram, since svm is using HIK)
                op.NHistogramBins.setValue(binSize)
                # set up classification method
                op.DetectionMethod.setValue('svm')
                # load list of positive examples (F_p) available for the training procedure
                # set of positive examples P = {(I_1,B_1), ... ,(I_n, B_n)}
                op.positive_TrainingHistograms.setValue(np.vstack(positive_train_histograms))
                # set of negative examples N = {J_1, ... , J_m}
                op.negative_TrainingHistograms.setValue(np.vstack(negative_train_histograms))
                op.train(force=True)
                # save detector
                try:
                    if detfile is None:
                        with tempfile.NamedTemporaryFile(
                                suffix='.pkl', prefix='detector_',
                                delete=False) as f:
                            f.write(op.dumps())
                    else:
                        with open(detfile, 'w') as f:
                            logger.info(
                                "Detector written to {}".format(f.name))
                            f.write(op.dumps())

                    logger.info(
                        "Detector written to {}".format(f.name))
                except Exception as e:
                    logger.error("==== BEGIN DETECTOR DUMP ====")
                    logger.error(op.dumps())
                    logger.error("==== END DETECTOR DUMP ====")
                    logger.error(str(e))

    logger.info(
        "Finished training script ({})".format(
            time.strftime("%Y-%m-%d %H:%M")))

    t_stop = time.time()

    logger.info("Duration: {}".format(
        time.strftime(
            "%Hh, %Mm, %Ss", time.gmtime((t_stop-t_start) % (24*60*60)))))
    if (t_stop-t_start) >= 24*60*60:
        logger.info(" and %d days!" % int(t_stop-t_start) // (24*60*60))
