import sys
import os
from climin.initialize import bound_spectral_radius
import numpy as np
import gnumpy as gp
import time
import theano
import gc
import ctypes
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

if sys.platform == 'nt':
    import msvcrt

############################################################################
# Misc utilities
############################################################################

def get_key():
    if sys.platform == 'win32':
        import msvcrt
        if msvcrt.kbhit():
            return msvcrt.getch()
        else:
            return None
    else:
        return None


def get_randseed():
    return int(time.time())


def cuda_device_reset():
    gc.collect()
    cuda_dll = ctypes.WinDLL("cudart64_55.dll")
    cuda_dll.cudaDeviceReset()


def steps(start, end, step):
    while start <= end:
        yield start
        start += step


def random_matrix_with_spectral_radius(size, std=1, bound=1.2):
    """
    Generates a randomly initialized, square matrix with bound spectral radius.
    :param size: size of matrix
    :type size: int
    :param bound: spectral radius bound
    :return: ndarray of shape (size, size)
    """
    mat = np.random.normal(0, std, size=(size, size))
    bound_spectral_radius(mat, bound=bound)
    return mat

def multifig(height_ratios=[1], figsize=(15, 15)):
    plt.figure(figsize=figsize)
    return GridSpec(len(height_ratios), 1, height_ratios=height_ratios)


############################################################################
# Utility functions operating on arrays/lists
############################################################################

def isfinite(x):
    if not isinstance(x, np.ndarray):
        return x.all_real()
    else:
        return np.all(np.isfinite(x))


def get_2d_meshgrid(x_min, x_max, x_num, y_min, y_max, y_num, as_matrix=False):
    """
    Returns either two vectors with all combinations of x_num and y_num values
    in [x_min, x_max] and [y_min, y_max] or a matrix with two rows, where
    the first corresponds to the x vector and the second to the y vector
    """
    assert x_min <= x_max, 'x_max must be larger or equal x_min'
    assert y_min <= y_max, 'y_max must be larger or equal y_min'
    assert x_num > 0 and y_num > 0, 'x_num and y_num must be positive'
    x = np.linspace(x_min, x_max, num=x_num)
    y = np.linspace(y_min, y_max, num=y_num)
    # leads to the same order as a nested for loop iterating first over x and
    # then over y (therefore switched here)
    y, x = np.meshgrid(x, y)
    x = x.flatten()
    y = y.flatten()
    if as_matrix:
        x = x.reshape((x.size,))
        y = y.reshape((y.size,))
        matrix = np.ones((2, len(x)))
        matrix[0, :] = x
        matrix[1, :] = y
        return matrix
    else:
        return x, y


def to_1hot(data, max_value):
    """
    Converts a one-dimensional data sequence to one-hot encoding.
    :param data: data[smpl] - integer data array to convert
    :param max_value: maximum value
    :return: onehot[value, smpl]
    """
    n_samples = data.shape[0]
    onehot = np.zeros((max_value + 1, n_samples))
    for smpl, val in enumerate(data):
        val = int(val)
        assert 0 <= val <= max_value
        onehot[val, smpl] = 1.0
    return onehot


def from_1hot(onehot):
    """
    Converts a one-hot encoded data sequence into a one-dimensional data sequence.
    :param onehot: onehot[value, smpl]
    :return: data[smpl]
    """
    return np.argmax(onehot, axis=0)


def sample_list_to_array(sample_list):
    """
    Joins a list of sample arrays into one joint array.
    :param sample_list: A list of arrays. Each array corresponds to one sample and the
                        last dimension is the step index.
    :return: An array, where the last dimension is the sample index and the second-last dimension is the step index.
    """
    other_dims = sample_list[0].shape[0:-1]
    max_steps = max([smpl.shape[-1] for smpl in sample_list])
    dims = other_dims + (max_steps, len(sample_list))

    ary = np.zeros(dims, dtype=sample_list[0].dtype)
    for idx, smpl in enumerate(sample_list):
        ary[..., 0:smpl.shape[-1], idx] = smpl
    return ary


def combine_sample_steps(n_steps, data):
    """
    Combines all steps from all samples of the given data.
    :param n_steps: n_steps[smpl] - number of the steps in the given sample.
    :param data: data[..., step, smpl] - data to combine
    :return: combined[..., smpl] - all valid steps from all samples concatenated
    """
    n_steps = np.asarray(n_steps)
    data = np.asarray(data)

    n_samples = n_steps.shape[0]
    assert data.shape[-1] == n_samples

    n_total_steps = np.sum(n_steps)
    shp = data.shape[0:-2]
    combined = np.zeros(shp + (n_total_steps,), dtype=data.dtype)

    pos = 0
    for smpl in range(n_samples):
        ns = n_steps[smpl]
        combined[..., pos : pos + ns] = data[..., 0:ns, smpl]
        pos += ns

    return combined


def divide_sample_steps(n_steps, combined):
    """
    Reverses the combination of steps of all samples done by the combine_sample_steps function.
    :param n_steps: n_steps: n_steps[smpl] - number of the steps in the given sample.
    :param combined: combined[..., smpl] - all valid steps from all samples concatenated
    :return: data: data[..., step, smpl] - data with separate dimensions for step and sample
    """
    n_samples = n_steps.shape[0]
    n_max_steps = np.max(n_steps)
    shp = combined.shape[0:-1]
    data = np.zeros(shp + (n_max_steps, n_samples), dtype=combined.dtype)

    pos = 0
    for smpl in range(n_samples):
        ns = n_steps[smpl]
        data[..., 0:ns, smpl] = combined[..., pos : pos + ns]
        pos += ns
    assert pos == combined.shape[-1]

    return data


############################################################################
# Theano modes
############################################################################

class PrintEverythingMode(theano.Mode):
    def __init__(self):
        def print_eval(i, node, fn):
            print '<' * 50
            print i, node, [input[0] for input in fn.inputs],
            fn()
            print [output[0] for output in fn.outputs]
            print '>' * 50
        wrap_linker = theano.gof.WrapLinkerMany(
            [theano.gof.OpWiseCLinker()], [print_eval])
        super(PrintEverythingMode, self).__init__(
            wrap_linker, optimizer='fast_compile')


class WarnNaNMode(theano.Mode):
    def __init__(self):
        def print_eval(i, node, fn):
            fn()
            for i, inpt in enumerate(fn.inputs):
                try:
                    if np.isnan(inpt[0]).any():
                        print 'nan detected in input %i of %s' % (i, node)
                        import pdb
                        pdb.set_trace()
                except TypeError:
                    print 'could not check for NaN in:', inpt
        wrap_linker = theano.gof.WrapLinkerMany(
            [theano.gof.OpWiseCLinker()], [print_eval])
        super(WarnNaNMode, self).__init__(
            wrap_linker, optimizer='fast_compile')


class Output2Floats(object):
    """
    Needed for Theano's scalar ops with two outputs to avoid using
    scalar.upgrade_to_float or upcast_out (which both always return one value)
    """
    def __new__(self, *types):
        type = types[0]
        if type in theano.scalar.float_types:
            return [type, type]
        else:
            return [theano.config.floatX, theano.config.floatX]
