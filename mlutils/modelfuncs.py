import numpy as np
from mlutils.gpu import post, gather


class ModelFuncs(object):

    def __init__(self, model, dataset):
        self.model = model
        self.dataset = dataset
        self._minibatch_idx = 0

        self._minibatch_idx = 0
        self._minibatch = None
        self.minibatch_idx = 0

    @property
    def minibatch(self):
        return self._minibatch

    @property
    def minibatch_idx(self):
        """The index of the minibatch to expose under the minibatch property."""
        return self._minibatch_idx

    @minibatch_idx.setter
    def minibatch_idx(self, value):
        self._minibatch_idx = int(value)
        self._minibatch = self.dataset.trn.minibatch(self._minibatch_idx)

    def next_minibatch(self):
        """
        Loads the next minibatchs.
        :return: specifies if the minibatch index has warped around (iteration finished)
        """
        if self.minibatch_idx == self.dataset.trn.n_minibatches - 1:
            self.minibatch_idx = 0
            return True
        else:
            self.minibatch_idx += 1
            return False

    @property
    def ps(self):
        """The parameterset of the model."""
        return self.model.ps

    def init_parameters(self):
        """Initializes the parameteres of the ParameterSet of the model close to zero."""
        self.ps.data[:] = post(np.random.normal(0, 0.01, size=self.model.ps.data.shape))

    def load_parameters(self, filename):
        """Loads the parameters of the ParameterSet of the model from the given file."""
        self.ps.data[:] = post(np.load(filename)['ps'])

    def save_parameters(self, filename):
        """Saves the parameters of the ParameterSet of the model to the given file."""
        np.savez_compressed(filename, ps=gather(self.ps.data))

