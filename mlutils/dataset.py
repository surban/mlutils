from math import ceil
import numpy as np
from mlutils.gpu import post, gather


class Dataset(object):
    """A dataset consisting of training, test and validation set."""

    def __init__(self, filename_or_ds, fractions=[0.8, 0.1, 0.1], minibatch_size=100, pad_data=False, seed=1,
                 split_function=None, preprocessing_function=None, with_all=False, force_cpu=False):
        """
        Loads a dataset from a .npz file and partitions it into training, test and validation set.
        :param filename_or_ds: file that contains the dataset or
                               a dict that contains (string, ndarray)-pairs
        :param fractions: split fractions for training, validation, test
        :param minibatch_size: size of a minibatch
                               (if None then one minibatch consisting of the whole dataset is created)
        :param pad_data: if True, data is padded so that all variables have the same size
        :param seed: random seed for splitting dataset into training, validation and test set
        :param split_function: a function that takes the dataset dictionary as input and returns
                               (training indices, validation indices, test indices)
        :param preprocessing_function: a function that takes the dataset dictionary as input and returns
                                       a preprocessed version of it
        :param with_all: if True, then self.all contains all samples of the dataset (uses additional memory)
        :param force_cpu: forces all data to be stored in the CPU memory and exposed as numpy arrays
        """
        if isinstance(filename_or_ds, dict):
            self._ds_source = 'data'
            self._ds_data = filename_or_ds
        else:
            self._ds_source = 'file'
            self._ds_filename = filename_or_ds
        self._fractions = fractions
        if minibatch_size is not None:
            self._minibatch_size = int(minibatch_size)
        else:
            self._minibatch_size = None
        self._pad_data = pad_data
        self._seed = seed
        self._with_all = with_all
        self._force_cpu = force_cpu

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
        if 'meta_splits' in ds:
            splits = np.asarray(ds['meta_splits'], dtype=int)
            if not (np.all(0 <= splits) and np.all(splits <= self.n_samples)):
                raise ValueError("meta_splits out of sample range")
            if not splits[0] <= splits[1]:
                raise ValueError("meta_splits[0] must be smaller or equal to meta_splits[1]")
            idx_trn = np.arange(0, splits[0])
            idx_val = np.arange(splits[0], splits[1])
            idx_tst = np.arange(splits[1], self.n_samples)
            print "Using dataset defined training/validation/test partitions."
        elif 'meta_idx_trn' in ds or 'meta_idx_val' in ds or 'meta_idx_tst' in ds:
            if not ('meta_idx_trn' in ds and 'meta_idx_val' in ds and 'meta_idx_tst' in ds):
                raise ValueError("meta_idx_trn, meta_idx_val, meta_idx_tst must be specified together")
            idx_trn = np.asarray(ds['meta_idx_trn'])
            idx_val = np.asarray(ds['meta_idx_val'])
            idx_tst = np.asarray(ds['meta_idx_tst'])
            print "Using dataset defined training/validation/test samples."
        else:
            old_rng = np.random.get_state()
            np.random.seed(self._seed)
            idx_trn, idx_val, idx_tst = self._splits(self.n_samples, self._fractions)
            np.random.set_state(old_rng)
        if 'meta_use_training_as_validation' in ds and ds['meta_use_training_as_validation']:
            print "Using training set as validation set."
            idx_val = idx_trn
        return idx_trn, idx_val, idx_tst

    def _load(self):
        # load or verify data
        if self._ds_source == 'file':
            ds = np.load(self._ds_filename)
        elif self._ds_source == 'data':
            ds = dict(self._ds_data)
            for key, value in ds.iteritems():
                ds[key] = np.asarray(value)
                if not isinstance(key, basestring):
                    raise TypeError("passed dataset must consist of (string, ndarray)-pairs. violated by key %s" % \
                                    str(key))

        # do preprocessing
        if self._preprocessing_function:
            ds = self._preprocessing_function(dict(ds))

        # extract number of samples and metadata
        self.n_samples = None
        for key, val in ds.iteritems():
            if key.startswith('meta_'):
                setattr(self, key, val)
            else:
                if self.n_samples is None:
                    self.n_samples = val.shape[-1]
                    n_samples_set_by = key
                elif val.shape[-1] != self.n_samples:
                    raise ValueError("dataset contains arrays with different number of samples (last dimension): "
                                     "n_samples=%d (from %s) but variable %s has shape %s" % \
                                     (self.n_samples, n_samples_set_by, key, str(val.shape)))

        # perform split
        idx_trn, idx_val, idx_tst = self._split_function(ds)
        idx_trn = np.asarray(idx_trn, dtype=int)
        idx_val = np.asarray(idx_val, dtype=int)
        idx_tst = np.asarray(idx_tst, dtype=int)

        # partitions
        self.trn = self.Paratition(ds, idx_trn, self._minibatch_size, self._pad_data, self._force_cpu)
        """Training set partition"""
        self.val = self.Paratition(ds, idx_val, self._minibatch_size, self._pad_data, self._force_cpu)
        """Validation set partition"""
        self.tst = self.Paratition(ds, idx_tst, self._minibatch_size, self._pad_data, self._force_cpu)
        """Test set partition"""
        if self._with_all:
            self.all = self.Paratition(ds, np.arange(self.n_samples), self._minibatch_size, self._pad_data,
                                       self._force_cpu)
            """All data"""
        else:
            self.all = None

    def print_info(self):
        """Prints info about this dataset"""
        n_bytes = self.trn.n_bytes + self.val.n_bytes + self.tst.n_bytes
        if self.all:
            n_bytes *= 2
        if self._ds_source == 'file':
            src = self._ds_filename
        elif self._ds_source == 'data':
            src = "<in-memory>"
        print "Dataset: %s" % src
        print "         (%d samples: %d training, %d validation, %d test, %.2f MB)" % \
              (self.n_samples, self.trn.n_samples, self.val.n_samples, self.tst.n_samples,
               n_bytes / float(2 ** 20))

    class Paratition(object):
        """A dataset partition (train / validation / test).
        Records from the dataset .npz file are exposed as members."""
        def __init__(self, ds, idx, minibatch_size, pad_data, force_cpu):
            self._keys = [k for k in ds.keys() if not k.startswith('meta_')]
            self._minibatch_size = minibatch_size
            self.n_samples = len(idx)
            self.n_bytes = 0

            for key in self._keys:
                data = ds[key][..., idx]
                if pad_data:
                    data = self._pad(data, minibatch_size)
                self.n_bytes += data.nbytes
                if not force_cpu:
                    data = post(data)
                setattr(self, key, data)

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
            if self._minibatch_size is not None:
                return Dataset.Minibatch(self, n * self._minibatch_size, (n + 1) * self._minibatch_size)
            else:
                return Dataset.Minibatch(self, 0, self.n_samples)

        @property
        def n_minibatches(self):
            """Number of minibatches"""
            if self._minibatch_size is not None:
                return int(ceil(self.n_samples / float(self._minibatch_size)))
            else:
                return 1

        def gather(self):
            return Dataset.GatheredPartition(self)

        def __getitem__(self, item):
            return getattr(self, item)

        def __contains__(self, item):
            return item in self._keys

        def iterkeys(self):
            return iter(self._keys)

        def iteritems(self):
            for key in self._keys:
                yield key, getattr(self, key)

    class GatheredPartition(object):
        def __init__(self, partition):
            self._keys = partition._keys
            for key in self._keys:
                data = gather(getattr(partition, key))
                setattr(self, key, data)

        def __getitem__(self, item):
            return getattr(self, item)

        def __contains__(self, item):
            return item in self._keys

        def iterkeys(self):
            return iter(self._keys)

        def iteritems(self):
            for key in self._keys:
                yield key, getattr(self, key)

    class Minibatch(object):
        """Minibatch of a dataset partition (train / validation / test).
        Records from the dataset .npz file are exposed as members."""

        def __init__(self, partition, b, e):
            if e > partition.n_samples:
                e = partition.n_samples
            for key in partition._keys:
                setattr(self, key, getattr(partition, key)[..., b:e])
            self._keys = partition._keys

        def __getitem__(self, item):
            return getattr(self, item)

        def __contains__(self, item):
            return item in self._keys

        def iterkeys(self):
            return iter(self._keys)

        def iteritems(self):
            for key in self._keys:
                yield key, getattr(self, key)


class DictPartition(object):

    def __init__(self, data):
        assert isinstance(data, dict)
        self._data = data
        for key, data in self._data.iteritems():
            setattr(self, key, data)

    def __getitem__(self, item):
        return self._data[item]

    def __contains__(self, item):
        return item in self._data

    def iterkeys(self):
        return self._data.iterkeys()

    def iteritems(self):
        return self._data.iteritems()

