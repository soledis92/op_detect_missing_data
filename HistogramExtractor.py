import numpy
import math

"""
histogram extractor (sliding window approach):
Input: * volume: 3D-data has to be 'C'-order (z,y,x)
       * cell_length: length of a squared cell in the grid
       * number_of_histogram_bins:

Output: * array of cells + corresponding histogram + root-location of the cell
            -> histogram is found in (z, y, x, :, 0)
            -> root-location is found in (z, y, x, 0, :) as (z_r, y_r, x_r)
"""


class HistogramExtractor(object):
    def __init__(self, volume, cell_length, number_of_histogram_bins, percentage_of_overlap):
        self._volume = volume
        self._cell_length = cell_length
        self._n_bins = number_of_histogram_bins
        self._overlap = percentage_of_overlap
        self._root_locations = []
        self._histograms = {}

    def _sanity_checks(self):
        assert len(self._volume.shape) == 3, "Volume must be 3d data"
        # check if evenly devideable:
        assert self._volume.shape[1] % self._cell_length == 0, "Volume y-axis must be evenly devideable by cell"
        assert self._volume.shape[2] % self._cell_length == 0, "Volume x-axis must be evenly devideable by cell"

    def _compute_root_locations(self):
        # root locations depend on the cell-size and the intended percentage of overlap
        x1 = 0
        x2 = 1
        y1 = 0
        y2 = 0
        while True:
            if _percentage_of_intersection(x1, y1, x2 + 1, y2, self._cell_length) >= self._overlap:
                x2 += 1
            else:
                break

        # distance needed between 2 root locations respecting the intended overlap
        distance = abs(x2-x1)
        # walk through all slices (z-axis of given volume)
        for z in range(self._volume.shape[0] + 1):
            # walk in y-direction, using steps of "distance"
            for y in range(self._cell_length/2, self._volume.shape[1] - self._cell_length/2 + 1, distance):
                # walk in x-direction, using steps of "distance"
                for x in range(self._cell_length/2, self._volume.shape[2] - self._cell_length/2 + 1, distance):
                    self._root_locations.append((z, y, x))
                self._root_locations.append((z, y, self._volume.shape[2] - self._cell_length/2))
            self._root_locations.append(
                (z, self._volume.shape[1] - self._cell_length/2, self._volume.shape[2] - self._cell_length/2))
            for x in range(self._cell_length/2, self._volume.shape[2] - self._cell_length/2 + 1, distance):
                self._root_locations.append((z, self._volume.shape[1] - self._cell_length/2, x))

    def _compute_histogram_to_dict(self, z_root, y_root, x_root):
        # check if window is within volume:
        x_min = x_root - self._cell_length / 2
        x_max = x_root + self._cell_length / 2
        y_min = y_root - self._cell_length / 2
        y_max = y_root + self._cell_length / 2
        z = z_root
        try:
            self._histograms[(z_root, y_root, x_root)] =\
                numpy.histogram(self._volume[z, y_min:y_max, x_min:x_max], self._n_bins, density = True)[0]
        except IndexError:
            # if cell is out of bounds, the histogram won't be computed.
            # The corresponding labeled pixel is discarded
            pass

    def getHistograms(self):
        self._compute_root_locations()
        # compute histogram for each root location
        for position in self._root_locations:
            self._compute_histogram_to_dict(position[0], position[1], position[2])
        # sort out the bad histograms (with nan or inf values)
        for sample in self._histograms.items():
            if numpy.any(numpy.isnan(sample[1])):
                del self._histograms[sample[0]]
            elif numpy.any(numpy.isinf(sample[1])):
                del self._histograms[sample[0]]
        return self._histograms.items()


def _percentage_of_intersection(x1, y1, x2, y2, a):
    # calculates the percentage of 2 squares with shape a**2, located at (x1, y1) and (x2, y2)
    #  d_x > 0
    if x1 > x2:
        if abs(x2-x1) >= a:
            return 0
        # d_y > 0
        if y1 > y2:
            if abs(y2-y1) >= a:
                return 0
            return math.fabs((x2 - x1 + a) * (y2 - y1 + a)) / (a * a)
        # d_y < 0
        elif y1 < y2:
            if abs(y1-y2) >= a:
                return 0
            return math.fabs((x2 - x1 + a) * (y2 - y1 - a)) / (a * a)
        # d_y = 0
        else:
            return 1 - math.fabs(x1-x2)/a
    # d_x < 0
    elif x1 < x2:
        if abs(x1-x2) >= a:
            return 0
        # d_y > 0
        if y1 > y2:
            if abs(y2-y1) >= a:
                return 0
            return math.fabs((x2 - x1 - a) * (y2 - y1 + a)) / (a * a)
        # d_y < 0
        elif y1 < y2:
            if abs(y1-y2) >= a:
                return 0
            return math.fabs((x2 - x1 - a) * (y2 - y1 - a)) / (a * a)
        # d_y = 0
        else:
            return 1 - math.fabs(x2 - x1) / a
    # d_x = 0
    else:
        # d_y > 0
        if y1 > y2:
            if abs(y2-y1) >= a:
                return 0
            return 1 - math.fabs(y2 - y1) / a
        # d_y < 0
        elif y1 < y2:
            if abs(y1-y2) >= a:
                return 0
            return 1 - math.fabs(y1 - y2) / a
        # d_y = 0
        else:
            return 100


########################################################################################################################
'''
histgram extractor:
as add on to the above one it takes labeled pixels as root-locations, while the above one just grids the complete
image into cells and doesn't use specific root locations
    -> the above one is only usefull for unlabeled data (background or prediction)
    -> faster but less negative examples
'''


class HistogramExtractorFromLabels(object):
    def __init__(self, volume, labels, which_label, cell_length, number_of_histogram_bins,
                 maximum_number_of_roots = None):
        self._volume = volume
        self._labels = labels
        self._label = which_label
        self._max_roots = maximum_number_of_roots
        self._free_root_locations = maximum_number_of_roots
        self._cell_length = cell_length
        self._n_bins = number_of_histogram_bins
        self._root_locations = []
        self._samples = {}
        self._step_size = 5

    def getHistograms(self):
        self._sanity_checks()
        self._compute_root_locations()
        print "number of roots before sorting out bad ones: ", len(self._root_locations)
        self._compute_all_histograms()
        # sort out the bad ones (histograms wih any values == nan | inf)
        for sample in self._samples.items():
            if numpy.any(numpy.isnan(sample[1])):
                del self._samples[sample[0]]
            elif numpy.any(numpy.isinf(sample[1])):
                del self._samples[sample[0]]
        return self._samples.items()

    def _sanity_checks(self):
        assert len(self._volume.shape) == 3, "Volume must be 3D, with axis 'zyx'"
        assert self._volume.shape == self._labels.shape, "labels must be same shape as volume"

    def _compute_root_locations(self):
        # each labeled pixel will be a root-location for a cell
        if self._label == 1:
            self._step_size = self._cell_length/2
        for z in range(self._labels.shape[0]):
            for y in range(self._cell_length, self._labels.shape[1], self._step_size):
                for x in range(self._cell_length, self._labels.shape[2], self._step_size):
                    if self._labels[z, y, x] == self._label:
                        self._root_locations.append((z, y, x))
            # restrict number of roots to max_roots
            if self._max_roots < len(self._root_locations):
                self._root_locations = self._root_locations[:self._max_roots]
                break

    def _compute_histogram_to_dict(self, z_root, y_root, x_root, background = False):
        # check if window is within volume:
        x_min = x_root - self._cell_length / 2
        x_max = x_root + self._cell_length / 2
        y_min = y_root - self._cell_length / 2
        y_max = y_root + self._cell_length / 2
        z = z_root
        try:
            if background:
                sum_over_labels = numpy.sum(self._labels[z, y_min:y_max, x_min:x_max])
                if sum_over_labels > self._cell_length**2:
                    # there is a defect in this patch -> not taken for training
                    pass
                else:
                    self._samples[(z_root, y_root, x_root)] =\
                        numpy.histogram(self._volume[z, y_min:y_max, x_min:x_max], self._n_bins, density = True)[0]
            else:
                self._samples[(z_root, y_root, x_root)] =\
                    numpy.histogram(self._volume[z, y_min:y_max, x_min:x_max], self._n_bins, density = True)[0]
        except IndexError:
            print "index error occours!"
            # if cell is out of bounds, the histogram won't be computed.
            # The corresponding labeled pixel is discarded
            pass

    def _compute_all_histograms(self):
        if self._label == 1:
            for position in self._root_locations:
                self._compute_histogram_to_dict(position[0], position[1], position[2], background = True)
        else:
            for position in self._root_locations:
                self._compute_histogram_to_dict(position[0], position[1], position[2])


class HistogramExtractorFullImage(object):
    def __init__(self, volume, number_of_histogram_bins):
        self._volume = volume
        self._histograms = {}
        self._n_bins = number_of_histogram_bins

    def _compute_histograms_to_dict(self):
        for z in range(self._volume.shape(2)):
            self._histograms[(z, 0, 0)] = numpy.histogram(self._volume[z, :, :], self._n_bins, density = True)[0]

    def _sanity_checks(self):
        assert len(self._volume.shape) == 3, "Volume must be 3D, with axis 'zyx'"

    def getHistograms(self):
        self._sanity_checks()
        self._compute_histograms_to_dict()
        # sort out the bad histograms (with nan or inf values)
        for sample in self._histograms.items():
            if numpy.any(numpy.isnan(sample[1])):
                del self._histograms[sample[0]]
            elif numpy.any(numpy.isinf(sample[1])):
                del self._histograms[sample[0]]

        return self._histograms.items()
