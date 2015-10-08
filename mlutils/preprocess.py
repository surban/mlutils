from __future__ import division
import numpy as np
from mlutils.misc import combine_sample_steps, divide_sample_steps


def pca_white(data, n_components=None, return_axes=False):
    """
    Performs PCA whitening of the data.
    :param data: data[feature, smpl] - input data
    :param n_components: If specified, limits the number of principal components to keep.
    :param return_axes: If True, this function additionaly returns the PCA variances and corresponding axes.
    :return: whitened[feature, smpl] - PCA whitened data.
             For every feature the mean over the samples is zero and the variance is one.
             The covariance matrix is diagonal.
    """
    data = np.asarray(data)
    n_samples = data.shape[1]

    # center data on mean
    means = np.mean(data, axis=1)
    centered = data - means[:, np.newaxis]

    # calculate principal axes and transform data into that coordinate system
    cov = np.dot(centered, centered.T) / n_samples
    variances, axes = np.linalg.eigh(cov)
    sort_idx = np.argsort(variances)[::-1]
    variances = variances[sort_idx]
    axes = axes[:, sort_idx]
    if n_components is not None:
        variances = variances[0:n_components]
        axes = axes[:, 0:n_components]
    pcaed = np.dot(axes.T, centered)

    # scale axes so that each has unit variance
    whitened = np.dot(np.diag(np.sqrt(1.0 / variances)), pcaed)

    if return_axes:
        return whitened, variances, axes
    else:
        return whitened


def zca(data):
    """
    Performs zero phase component analysis (ZCA) of the data.
    :param data: data[feature, smpl] - input data
    :return: zcaed[feature, smpl] - ZCA whitened data.
             For every feature the mean over the samples is zero and the variance is one.
             The covariance matrix is diagonal.
             Furhtermore zcaed is most similar to data in the least square sense, i.e. (zcaed - data)**2 is minimized
             with the constraint that above properties are satisfied.
    """
    # get PCA whitened data
    whitened, variances, axes = pca_white(data, return_axes=True)

    # rotate back into original coordinate system
    zcaed = np.dot(axes, whitened)

    return zcaed


def for_step_data(func):
    """
    Builds a preprocessing function for step data, i.e. data of the form data[feature, step, smpl].
    The built function expects the number of steps for each sample as first argument and the data as second argument.
    It returns the preprocesd data in the same shape as the input data.
    :param func: the preprocessing function
    :return: preprocessing_func(n_steps[smpl], data[feature, step, smpl])
    """
    def sd_func(n_steps, data, *args, **kwargs):
        data_c = combine_sample_steps(n_steps, data)
        ret = func(data_c, *args, **kwargs)
        if isinstance(ret, tuple):
            data_pc = ret[0]
        else:
            data_pc = ret
        data_p = divide_sample_steps(n_steps, data_pc)
        if isinstance(ret, tuple):
            return (data_p,) + ret[1:]
        else:
            return data_p
    return sd_func


