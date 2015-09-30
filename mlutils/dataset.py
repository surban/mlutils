from math import ceil
import numpy as np
from mlutils.gpu import post


class Dataset(object):
    """A dataset consisting of training, test and validation set."""

    def __init__(self, filename, fractions=[0.8, 0.1, 0.1], minibatch_size=100, pad_data=False, seed=1,
                 split_function=None, preprocessing_function=None, with_all=False):
        """
        Loads a dataset from a .npz file and partitions it into training, test and validation set.
        :param filename: file that contains the dataset
        :param fractions: split fractions for training, validation, test
        :param minibatch_size: minibatch set
        :param pad_data: if True, data is padded so that all variables have the same size
        :param seed: random seed for splitting dataset into training, validation and test set
        :param split_function: a function that takes the dataset dictionary as input and returns
                               (training indices, validation indices, test indices)
        :param preprocessing_function: a function that takes the dataset dictionary as input and returns
                                       a preprocessed version of it
        :param with_all: if True, then self.all contains all samples of the dataset (uses additional memory)
        """
        self._filename = filename
        self._fractions = fractions
        self._minibatch_size = int(minibatch_size)
        self._pad_data = pad_data
        self._seed = seed
        self._with_all = with_all

        if split_function is not None:
            self._split_function = split_function
        else:
            self._split_function = self._default_splits
        self._preprocessing_function = preprocessing_function

        self._load()
        self.print_info()

    def _splits(self, total, fractions):
        fractions = np.asarray(fractions)
        tf = np.sum(fractions)
        counts = np.asarray(np.floor(fractions * total / float(tf)), dtype=int)
        counts[-1] = total - np.sum(counts[0:-1])
        cuts = np.concatenate(([0], np.cumsum(counts)))

        idx = np.random.choice(total, size=total, replace=False)
        splits = []
        for i in range(len(fractions)):
            splits.append(idx[cuts[i] : cuts[i+1]])
        return splits

    def _default_splits(self, ds):
        old_rng = np.random.get_state()
        np.random.seed(self._seed)
        idx_trn, idx_val, idx_tst = self._splits(self.n_samples, self._fractions)
        np.random.set_state(old_rng)
        return idx_trn, idx_val, idx_tst

    def _load(self):
        ds = np.load(self._filename)
        if self._preprocessing_function:
            ds = self._preprocessing_function(dict(ds))

        self.n_samples = ds[ds.keys()[0]].shape[-1]
        for key in ds.keys():
            if ds[key].shape[-1] != self.n_samples:
                raise ValueError("dataset contains arrays with different number of samples (last dimension)")

        # perform split
        idx_trn, idx_val, idx_tst = self._split_function(ds)
        idx_trn = np.asarray(idx_trn, dtype=int)
        idx_val = np.asarray(idx_val, dtype=int)
        idx_tst = np.asarray(idx_tst, dtype=int)

        # partitions
        self.trn = self.Paratition(ds, idx_trn, self._minibatch_size, self._pad_data)
        """Training set partition"""
        self.val = self.Paratition(ds, idx_val, self._minibatch_size, self._pad_data)
        """Validation set partition"""
        self.tst = self.Paratition(ds, idx_tst, self._minibatch_size, self._pad_data)
        """Test set partition"""
        if self._with_all:
            self.all = self.Paratition(ds, np.arange(self.n_samples), self._minibatch_size, self._pad_data)
            """All data"""
        else:
            self.all = None

    def print_info(self):
        """Prints info about this dataset"""
        n_bytes = self.trn.n_bytes + self.val.n_bytes + self.tst.n_bytes
        if self.all:
            n_bytes *= 2
        print "Dataset: %s  (%d samples: %d training, %d validation, %d test, %.2f MB)" % \
              (self._filename, self.n_samples, self.trn.n_samples, self.val.n_samples, self.tst.n_samples,
               n_bytes / float(2 ** 20))

    class Paratition(object):
        """A dataset partition (train / validation / test).
        Records from the dataset .npz file are exposed as members."""
        def __init__(self, ds, idx, minibatch_size, pad_data):
            self._keys = ds.keys()
            self._minibatch_size = minibatch_size
            self.n_samples = len(idx)
            self.n_bytes = 0

            for key in self._keys:
                data = ds[key][..., idx]
                if pad_data:
                    data = self._pad(data, minibatch_size)
                self.n_bytes += data.nbytes
                setattr(self, key, post(data))

        def _pad(self, data, multiple):
            n_smpls = data.shape[-1]
            if n_smpls % multiple == 0:
                return data
            else:
                paded_shape = data.shape
                paded_shape[-1] = (int(n_smpls / multiple) + 1) * multiple
                paded_data = np.zeros(paded_shape, dtype=data.dtype)
                paded_data[..., 0:n_smpls] = data[..., 0:n_smpls]
                return paded_data

        def minibatch(self, n):
            """
            Returns the n-th minibatch.
            :param n: the index of the minibatch
            :return: the n-th minibatch
            """
            n = int(n)
            if n < 0 or n > self.n_minibatches:
                raise ValueError("minibatch out of range")
            return Dataset.Minibatch(self, n * self._minibatch_size, (n + 1) * self._minibatch_size)

        @property
        def n_minibatches(self):
            """Number of minibatches"""
            return int(ceil(self.n_samples / self._minibatch_size))

    class Minibatch(object):
        """Minibatch of a dataset partition (train / validation / test).
        Records from the dataset .npz file are exposed as members."""

        def __init__(self, partition, b, e):
            if e > partition.n_samples:
                e = partition.n_samples
            for key in partition._keys:
                setattr(self, key, getattr(partition, key)[..., b:e])


