import numpy as np
from os.path import exists, join
from os import unlink
from mlutils.config import optimizer_from_cfg
from mlutils.dataset import Dataset
from mlutils.gpu import post, gather
from mlutils.parameterhistory import ParameterHistory


class ModelFuncs(object):
    """Base class for callable functions of a Theano model."""

    def __init__(self, model, cfg, dataset=None):
        self.model = model
        self.cfg = cfg
        if dataset is not None:
            self.dataset = dataset
        else:
            self.dataset = Dataset(self.cfg.dataset, minibatch_size=self.cfg.minibatch_size)

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

    def init_parameters(self, var=None):
        """Initializes the parameteres of the ParameterSet using a zero-mean normal distribution.
        :param var: Variance of the normal distribution.
                    If None, then the 'initialization_variance' parameter from the configuration is used.
        """
        if var is None:
            if 'initialization_variance' in dir(self.cfg):
                var = self.cfg.initialization_variance
            else:
                var = 0.01
        print "Initializing parameters with variance %f" % var
        self.ps.data[:] = post(np.random.normal(0, var, size=self.model.ps.data.shape))

    def load_parameters(self, filename):
        """Loads the parameters of the ParameterSet of the model from the given file."""
        self.ps.data[:] = post(np.load(filename)['ps'])

    def save_parameters(self, filename):
        """Saves the parameters of the ParameterSet of the model to the given file."""
        np.savez_compressed(filename, ps=gather(self.ps.data))

    def loss(self, pv, dataset):
        """Calculates the loss on the given dataset with the given parameteres.
        :param pv: parameter values
        :param dataset: dataset to calculate loss on
        """
        raise NotImplementedError("loss must be implemented for the specific model")

    def loss_grad(self, pv, dataset):
        """Calculates the gradient of the loss w.r.t. the parameters on the given dataset.
        :param pv: parameter values
        :param dataset: dataset to calculate loss on
        """
        raise NotImplementedError("loss_grad must be implemented for the specific model")

    def mb_loss(self, pv):
        """Calculates the loss on the current dataset with the given parameters.
        :param pv: parameter values"""
        return self.loss(pv, self.minibatch)

    def mb_loss_grad(self, pv):
        """Calculates the gradient of the loss on the current dataset with the given parameters.
        :param pv: parameter values"""
        return self.loss_grad(pv, self.minibatch)

    @property
    def trn_loss(self):
        """Loss on the whole training set using the current parameters."""
        return gather(self.loss(self.ps.data, self.dataset.trn))

    @property
    def val_loss(self):
        """Loss on the whole validation set using the current parameters."""
        return gather(self.loss(self.ps.data, self.dataset.val))

    @property
    def tst_loss(self):
        """Loss on the whole test set using the current parameters."""
        return gather(self.loss(self.ps.data, self.dataset.tst))

    def record_loss(self, history, iter):
        """
        Records the current losses in the specified ParameterHistory object
        and checks if training should be terminated.
        :param iter: the iteration number
        :param history: the ParameterHistory object that should store the loss
        :type history: ParameterHistory
        :return: true, if training should be terminated, false other.
        """
        history.add(iter, self.ps.data, self.trn_loss, self.val_loss, self.tst_loss)
        return history.should_terminate

    def generic_training(self, cfg_dir, checkpoint=None, checkpoint_handler=None, loss_record_interval=10,
                         max_missed_val_improvements=200, reset_termination_criteria=False):
        """
        Generic training procedure.
        :param cfg_dir: configuration directory
        :param checkpoint: checkpoint
        :param checkpoint_handler: checkpoint handler
        :param loss_record_interval: number of iterations between calculating and recording losses
        :param max_missed_val_improvements: maximum iterations without improvement of loss before training ist stopped
        :param reset_termination_criteria: resets the termination criteria after loading a checkpoint
        :return: ParameterHistory object of training
        """
        # create optimizer
        opt = optimizer_from_cfg(self.cfg, self.ps.data, self.mb_loss, self.mb_loss_grad)

        # initialize or restore checkpoint, if available
        if not checkpoint:
            self.init_parameters()
            his = ParameterHistory(cfg=self.cfg, state_dir=cfg_dir, max_iters=self.cfg.max_iters,
                                   max_missed_val_improvements=max_missed_val_improvements)
            iter = 0
        else:
            self.ps.data[:] = post(checkpoint['data'])
            his = checkpoint['his']
            his.state_dir = cfg_dir
            his.max_missed_val_improvements = max_missed_val_improvements
            iter = checkpoint['iter']

        # reset termination criteria if requested
        second_chance_file = join(cfg_dir, "2nd_chance")
        if exists(second_chance_file):
            print "Resetting termination criteria because %s is present" % second_chance_file
            reset_termination_criteria = True
            unlink(second_chance_file)
        if reset_termination_criteria:
            his.should_terminate = False

        # do training
        if not his.should_terminate:
            print "Training..."
            for sts in opt:
                if self.next_minibatch():
                    # iteration finished
                    iter += 1

                    # calculate losses
                    if iter % loss_record_interval == 0:
                        if self.record_loss(his, iter):
                            break

                # save checkpoint if necessary
                if checkpoint_handler and checkpoint_handler.requested:
                    checkpoint_handler.save(data=gather(self.ps.data), his=his, iter=iter)

        # save results and plot loss
        if checkpoint_handler:
            checkpoint_handler.save(data=gather(self.ps.data), his=his, iter=iter, explicit=True)
        his.finish()

        return his
