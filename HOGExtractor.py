import numpy


class HOGExtractor(object):
    """
    Extracts histograms of oriented gradients (HOG) from given volume.
    volume and labels should be same shape, 3D with axis: 'zyx'
    y-x-shape has to be evenly devideable by cellLength
    cells are quadraticly shaped

    Input:
        * volume: 3d array in c-order <-> 'zyx'
        * cellLength: int, length of one side of a cell. Cells must devide yx-plane of volume evenly
        * numberOfHistogramBins: int, define angular steps by chosing a histogram bin number: e.g. 8 = 45 deg
                                                                                                  360 = 1 deg
    Output:
        via
        * getOverAllCellHistograms
            returns dictionary of histograms of average orientation of cells,
            using average magnitude of each cell as the corresponding weight.
            Each histogram has the corresponding z_index as its key
        * getHOGArray
            returns cells as an array; shape: (z, y_cell, x_cell, number of bins of histogram(cell))
            the histograms for each cell is normalized (numpy.histogram(density = True))
            a HOG descriptor can easily be computed by applying different blocks to the HOGArray
            and computing the concatenated vector of the histogram-bins of all cells in a block.


    later on this class could be improved by only calculating cells around pre-defined label positions + parallelization
    """
    def __init__(self, volume, cellLength, numberOfHistogramBins):
        self.__volume = volume
        self.__cell_length = cellLength
        self.__number_of_cells = volume.shape[1] / self.__cell_length
        self.__n_bins = numberOfHistogramBins
        # gradient.shape = (volume.shape, (gradient_x, gradient_y, magnitude, direction))
        self._gradient = numpy.ndarray(
            (self.__volume.shape[0],
             self.__volume.shape[1],
             self.__volume.shape[2], 4), dtype = numpy.int8)
        # oriented_cells.shape =
        # (volume.shape[0], squared cellArray.shape, (average orientation of cell, average weight))
        self._oriented_cells = numpy.ndarray(
            (self.__volume.shape[0],
             self.__volume.shape[1] / self.__cell_length,
             self.__volume.shape[2] / self.__cell_length, 2), dtype = numpy.int64)
        # histogram_cells.shape = (volume.shape[0], squared cellArray.shape, orientation-weighted histogram of the cell)
        self.__histogram_cells = numpy.ndarray(
            (self.__volume.shape[0],
             self.__volume.shape[1] / self.__cell_length,
             self.__volume.shape[2] / self.__cell_length, self.__n_bins), dtype = numpy.int64)
        self.__overall_histograms = {}

# sanity checks
    def _sanity_checks(self):
        assert len(self.__volume.shape) == 3, "data must be 3D"
        assert self.__volume.shape[1] % self.__cell_length == 0
        assert self.__volume.shape[2] % self.__cell_length == 0

# histograms computed for each cell:
    def _compute_orient_weighted_cell_histograms_matrix(self):
        # compute an orientation histogram of all pixels in each cell.
        # Each pixel is weighted with its gradient magnitude.
        occuring_orientation_magnitude_pairs = {}
        # walk over all cells
        for z_cell in range(self._oriented_cells.shape[0]):
            for y_cell in range(self._oriented_cells.shape[1]):
                for x_cell in range(self._oriented_cells.shape[2]):
                    # walk over corresponding pixels
                    for y in range(y_cell, y_cell + self.__cell_length):
                        for x in range(x_cell, x_cell + self.__cell_length):
                            # append pair(orientation, magnitude) of current pixel
                            occuring_orientation_magnitude_pairs[self._gradient[z_cell, y, x, 3]] =\
                                self._gradient[z_cell, y, x, 2]
                    # compute histogram of current cell using pixel-gradient-magnitudes as weights
                    self.__histogram_cells[z_cell, y_cell, x_cell, :] =\
                        numpy.histogram(occuring_orientation_magnitude_pairs.keys(),
                                        bins = self.__n_bins,
                                        weights = occuring_orientation_magnitude_pairs.values(), density = True)[0]

# orientation + average first, then overall histogram:
    def _orient_cells(self):
        # calculates average orientation and average magnitude for each cell
        n_pixels = self.__cell_length * self.__cell_length
        # walk over all cells
        for z_cell in range(self._oriented_cells.shape[0]):
            for y_cell in range(self._oriented_cells.shape[1]):
                for x_cell in range(self._oriented_cells.shape[2]):
                    # walk over corresponding pixels
                    for y in range(y_cell, y_cell + self.__cell_length-1):
                        for x in range(x_cell, x_cell + self.__cell_length-1):
                            # z_cell can be used since z_cell == z (we only calculate cells in y-x-plane)
                            self._oriented_cells[z_cell, y_cell, x_cell, 0] += self._gradient[z_cell, y, x, 2]/n_pixels
                            self._oriented_cells[z_cell, y_cell, x_cell, 1] += self._gradient[z_cell, y, x, 3]/n_pixels

    def _compute_histogram_of_all_oriented_cells(self):
        # histogram is computed over the whole cell matrix.
        # Each cell already has an average orientation and an average magnitude.
        for z in range(self._oriented_cells.shape[0]):
            self.__overall_histograms[z] =\
                numpy.histogram(self._oriented_cells[z, :, :, 0],
                                bins = self.__n_bins,
                                weights = self._oriented_cells[z, :, :, 1])[0]

# gradient computing
    def _compute_gradient(self):
        # computing the gradient using the sobel operator
        # result will be stored in self._gradient as (volume.shape, gradient_x, gradient_y, magnitude, direction)
        assert self.__volume.shape[1] % 3 == 0, "volume isn't evenly devideable by 3: problem with 3x3 sobel-operator"
        for z in range(self.__volume.shape[0]):
            for i in range(self.__volume.shape[1] / 3):
                for j in range(self.__volume.shape[1] / 3):
                    mask = self.__volume[z, i * 3:(i + 1) * 3, j * 3:(j + 1) * 3]
                    self._gradient[z, i * 3 + 1, j * 3 + 1] = self.__sobel(mask)

    @staticmethod
    # simple implementation of 2D-sobel operator
    # mask should be a 3x3 matrix with pixel of interest as center
    def __sobel(mask):
        gradient_x = mask[0, 0] + 2 * mask[1, 0] + mask[2, 0] - mask[0, 2] - 2 * mask[1, 2] - mask[2, 2]
        gradient_y = mask[0, 0] + 2 * mask[0, 1] + mask[0, 2] - mask[2, 0] - 2 * mask[2, 1] - mask[2, 2]
        magnitude = abs(gradient_x) + abs(gradient_y)
        # direction in grad
        direction = numpy.arctan2(gradient_y, gradient_x) * (180/3.1415)
        return gradient_x, gradient_y, magnitude, direction

# API methods for HOGExtractor:
    def getOverAllCellHistograms(self):
        self._sanity_checks()
        self._compute_gradient()
        self._orient_cells()
        self._compute_histogram_of_all_oriented_cells()
        return self.__overall_histograms

    def getHOGArray(self):
        self._sanity_checks()
        self._compute_gradient()
        self._compute_orient_weighted_cell_histograms_matrix()
        return self.__histogram_cells
