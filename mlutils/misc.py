import sys
from climin.initialize import bound_spectral_radius
import numpy as np
import time
import theano
import theano.tensor as T
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
    max_steps = max([smpl.shape[-1] for smpl in sample_list]) + 1   # ensure one invalid sample between sequences
    dims = other_dims + (max_steps, len(sample_list))

    ary = np.zeros(dims, dtype=sample_list[0].dtype)
    for idx, smpl in enumerate(sample_list):
        ary[..., 0:smpl.shape[-1], idx] = smpl
    return ary

def sample_list_to_valid(sample_list):
    valid_list = [np.ones(smpl.shape[-1]) for smpl in sample_list]
    return sample_list_to_array(valid_list)

def sample_list_to_n_steps(sample_list):
    n_steps = [smpl.shape[-1] for smpl in sample_list]
    return np.asarray(n_steps, dtype=int)

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

def combine_valid_steps(valid, data):
    """
    Combines all steps from all samples of the given data.
    :param valid: valid[step, smpl] 
    :param data: data[..., step, smpl] - data to combine
    :return: combined[..., smpl] - all valid steps from all samples concatenated
    """
    return combine_sample_steps(valid_to_n_steps(valid), data)

def divide_sample_steps(n_steps, combined, n_max_steps=None):
    """
    Reverses the combination of steps of all samples done by the combine_sample_steps function.
    :param n_steps: n_steps: n_steps[smpl] - number of the steps in the given sample.
    :param combined: combined[..., smpl] - all valid steps from all samples concatenated
    :return: data: data[..., step, smpl] - data with separate dimensions for step and sample
    """
    n_samples = n_steps.shape[0]
    if n_max_steps is None:
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

def divide_into_sequences(last_step, combined):
    """
    Reverses the combination of steps of all samples done by the combine_sample_steps function.
    :param last_step: last_step[smpl]
    :param combined: combined[..., smpl] - all valid steps from all samples concatenated
    :return: data: data[..., step, smpl] - data with separate dimensions for step and sample
    """
    n_steps = last_step_to_n_steps(last_step)
    return divide_sample_steps(n_steps, combined)

def last_step_to_n_steps(last_step):
    """
    Reconstructs the number of steps of each sequence from a is-last-step vector.
    :param last_step: last_step[smpl]
    :return: n_steps[smpl]
    """
    n_steps = []
    ns = 0
    for ls in last_step:
        ns += 1
        if ls != 0:
            n_steps.append(ns)
            ns = 0
    return np.asarray(n_steps, dtype=int)

def flat_valid_to_n_steps(flat_valid):
    """
    Reconstructs the number of steps of each sequence from a flattened valid vector.
    There must be at least one invalid sample between two sequences.
    :param flat_valid: flat_valid[smpl]
    :return: n_steps[smpl]
    """
    n_steps = []
    ns = 0
    last_valid = True
    for v in flat_valid:
        if v != 0:
            ns += 1
            last_valid = True
        elif last_valid:
            n_steps.append(ns)
            ns = 0
            last_valid = False
    if last_valid:
        n_steps.append(ns)
    return np.asarray(n_steps, dtype=int)


def valid_to_n_steps(valid):
    """
    Counts the number of valid steps (for each sample).
    :param valid: valid[step, smpl] or valid[step]
    :returns: n_steps[smpl] or n_steps
    """
    if valid.ndim == 1:
        return np.count_nonzero(valid)
    elif valid.ndim == 2:
        n_samples = valid.shape[1]
        n_steps = np.zeros(n_samples, dtype=int)
        for smpl in range(n_samples):
            n_steps[smpl] = valid_to_n_steps(valid[:, smpl])
        return n_steps  
    else:
        raise ValueError("valid has wrong number of dimensions")  
        
def n_steps_to_valid(n_steps, max_steps=None):
    """
    Converts the number of valid steps (for each sample) into a validity vector.
    :param n_steps: n_steps[smpl] or n_steps
    :param max_steps: maximum number of steps (used as length for valid vector) 
    :returns: valid[step, smpl] or valid[step]
    """
    if isinstance(n_steps, (int, long)):
        if max_steps is None:
            max_steps = n_steps
        valid = np.zeros(max_steps)
        valid[0:n_steps] = 1
    else:
        if max_steps is None:
            max_steps = np.max(n_steps)
        n_samples = n_steps.shape[0]
        valid = np.zeros((max_steps, n_samples))
        for smpl in range(n_samples):
            valid[:, smpl] = n_steps_to_valid(n_steps[smpl], max_steps)
        return valid

def combine_valid_steps_of_dataset(ds):
    """
    Converts a dataset that has variables of the form var[..., step, smpl] to the 
    form var[..., smpl], i.e. each step becomes a separate sample.
    :param ds: dict of (name, value[..., step, smpl])
    :returns: dict of (name, value[..., smpl])
    """
    valid = ds['valid']
    n_steps = valid_to_n_steps(valid)
    n_samples = valid.shape[1]
    valid_steps, valid_smpls = np.nonzero(valid)
    n_flat_samples = len(valid_smpls)

    # flatten variables over steps and samples 
    flt_ds = {}
    for var, val in ds.iteritems():
        if var == 'valid' or var == 'n_steps':
            pass
        elif var.startswith('meta_'):
            flt_ds[var] = val
        else:
            # print "combining %s of shape %s" % (var, str(val.shape))
            flt_ds[var] = combine_valid_steps(valid, val)

    # map steps of samples to new sample indices
    smpl_becomes = {}
    pos = 0
    for smpl in range(n_samples):
        smpl_becomes[smpl] = []
        for step in range(n_steps[smpl]):
            smpl_becomes[smpl].append(pos)
            pos += 1

    # map data set partitions specified in dataset to new indices
    for mi in ['meta_idx_trn', 'meta_idx_val', 'meta_idx_tst']:
        flt_ds[mi] = reduce(lambda l, smpl: l + smpl_becomes[smpl], ds[mi], [])

    # mark last steps of each sequence to make division back into sequences possible
    last_step = np.zeros(n_flat_samples)
    for smpl in range(n_samples):
        last_step[smpl_becomes[smpl][-1]] = 1
    flt_ds['last_step'] = last_step

    return flt_ds

def divide_combined_dataset(flt_ds):
    """
    Divides a dataset that was processed using the combine_valid_combine_valid_steps_of_dataset function
    back into its original form.
    :param flt_ds: dict of (name, value[..., smpl])
    :return: dict of (name, value[..., step, smpl])
    """
    n_steps = last_step_to_n_steps(flt_ds['last_step'])

    ds = {}
    for var, val in flt_ds.iteritems():
        if var == 'last_step':
            pass
        elif var.startswith('meta_'):
            ds[var] = val
        else:
            ds[var] = divide_sample_steps(n_steps, val)

    ds['n_steps'] = n_steps
    ds['valid'] = n_steps_to_valid(n_steps)

    return ds

def divide_combined_dataset_var(ds, var):
    """
    Divides a variable from a dataset that was processed using the combine_valid_combine_valid_steps_of_dataset function
    back into its original form.
    :param ds: dict of (name, value[..., step, smpl]) (not the combined dataset)
    :param var: var[..., smpl]
    :return: var[..., step, smpl]
    """
    if 'n_steps' not in ds:
        raise ValueError("dataset must not be the combined version (apply divide_combined_dataset first)")
    return divide_sample_steps(ds['n_steps'], var)


############################################################################
# Theano intermediate printing
############################################################################


def _print_fn(op, xin, only_notfinite, slc):
    if only_notfinite:
        pos = np.transpose(np.nonzero(~np.isfinite(xin)))
        if pos.size > 0:
            print "!!!!!! non-finite %s at index " % op.message,
            for i in range(pos.shape[0]):
                print str(tuple(pos[i, :])) + ", ",
            print
    else:
        if slc is None:
            print "%s = \n%s" % (op.message, str(xin))
        else:
            slc = tuple((slice(None) if p is None else p for p in slc))
            print "%s[%s] = \n%s" % (op.message, str(slc), str(xin[slc]))


def print_node(caption, val, only_notfinite=False, slice=None):
    """
    Prints the value of a Theano node during evaluation.
    :param caption: name to display
    :param val: input node
    :param only_notfinite: if True, a list of non-finite indices is printed
    :param slice: optional slice of value to print
    :return: node with print side effect
    """
    return T.printing.Print(caption,
                            global_fn=lambda op, xin: _print_fn(op, xin,
                                                                only_notfinite=only_notfinite,
                                                                slc=slice))(val)


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
