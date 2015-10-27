import numpy as np
from os.path import exists, join
from os import unlink
from mlutils import xp
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

        self.set_constant_parameters_from_cfg()

    def set_constant_parameters_from_cfg(self):
        """
        Sets constant parameters in the ParameterSet from the configuration.
        """
        prefix = 'const_'
        for name in dir(self.cfg):
            if name.startswith(prefix):
                varname = name[len(prefix):]
                value = getattr(self.cfg, name)

                print "Constant paramter %s = %s" % (varname, repr(value))
                self.ps.constants[varname] = post(value)

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
        """The parameterset of the model.
        :type: ParameterSet
        """
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
        self.ps.restore_constants()

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
        if self.dataset.val.n_samples > 0:
            val_loss = self.val_loss
        else:
            val_loss = self.trn_loss
        if self.dataset.tst.n_samples > 0:
            tst_loss = self.tst_loss
        else:
            tst_loss = 0
        history.add(iter, self.ps.data, self.trn_loss, val_loss, tst_loss)
        return history.should_terminate

    def generic_training(self, cfg_dir, checkpoint=None, checkpoint_handler=None, loss_record_interval=10,
                         max_missed_val_improvements=200, iteration_gain=1.25, reset_termination_criteria=False,
                         desired_loss=None, initialize=True,
                         large_gradient_threshold=0.0, print_gradient_info=False, print_gradient=False,
                         print_parameters=False):
        """
        Generic training procedure.
        :param cfg_dir: configuration directory
        :param checkpoint: checkpoint
        :param checkpoint_handler: checkpoint handler
        :param loss_record_interval: number of iterations between calculating and recording losses
        :param max_missed_val_improvements: maximum iterations without improvement of loss before training ist stopped
        :param iteration_gain: If not set to 0, then training is performed up to iteration
                               (iteration_gain * iteration_of_last_improvement).
        :param reset_termination_criteria: resets the termination criteria after loading a checkpoint
        :param desired_loss: if specified, training is terminated with this loss is reached
        :param initialize: if True, the model parameters are initialized using the init_parameters method.
        :param large_gradient_threshold: if specified, a check for large gradient elements that exceed the
                                         given threshold is performed every iteration and they are printed.
        :param print_gradient_info: if True, this function prints diagnostic gradient information
        :param print_gradient: if True, this function prints the full gradient every minibatch
        :return: ParameterHistory object of training
        """

        # build gradient preprocessing chain
        grad_func = self.mb_loss_grad

        # gradient magnitude cap
        if 'gradient_cap' in dir(self.cfg) and self.cfg.gradient_cap is not None:
            grad_with_cap_orig_grad = grad_func
            def grad_with_cap(pv):
                g = grad_with_cap_orig_grad(pv)
                g_mag = xp.sqrt(xp.sum(g**2))
                if g_mag > self.cfg.gradient_cap:
                    print "gradient magnitude %f is being rescaled" % g_mag
                    g *= self.cfg.gradient_cap / g_mag
                return g
            grad_func = grad_with_cap

        # gradient element cap
        if 'gradient_element_cap' in dir(self.cfg) and self.cfg.gradient_element_cap is not None:
            grad_with_element_cap_orig_grad = grad_func
            def grad_with_element_cap(pv):
                g = grad_with_element_cap_orig_grad(pv)
                elems = xp.where(xp.abs(g) > self.cfg.gradient_element_cap)
                g[elems] = xp.sign(g[elems]) * self.cfg.gradient_element_cap
                return g
            grad_func = grad_with_element_cap

        # gradient of constants set to zero
        grad_without_const_orig_grad = grad_func
        def grad_without_const(pv):
            g = grad_without_const_orig_grad(pv)
            return self.ps.nullify_gradient_of_constants(g)
        grad_func = grad_without_const

        # create optimizer
        opt = optimizer_from_cfg(self.cfg, self.ps.data, self.mb_loss, grad_func)

        # initialize or restore checkpoint, if available
        if not checkpoint:
            if initialize:
                self.init_parameters()
            his = ParameterHistory(cfg=self.cfg, state_dir=cfg_dir, max_iters=self.cfg.max_iters,
                                   max_missed_val_improvements=max_missed_val_improvements,
                                   desired_loss=desired_loss, iteration_gain=iteration_gain)
            iter = 0
            # Record initial loss
            self.record_loss(his, iter)
        else:
            self.ps.data[:] = post(checkpoint['data'])
            his = checkpoint['his']
            his.state_dir = cfg_dir
            his.max_missed_val_improvements = max_missed_val_improvements
            his.desired_loss = desired_loss
            setattr(his, 'iteration_gain', iteration_gain)
            iter = checkpoint['iter']
            # start and endtimes in his should have the same length, this is
            # not the case in explicit auto-save checkpoints, therefore set the
            # end to the saved
            if len(his.start_time) != len(his.end_time):
                his.end_time.append(checkpoint['save_time'])
            his.start()

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
            self.ps.restore_constants()
            for sts in opt:
                gradient = None

                if large_gradient_threshold > 0:
                    gradient = gather(sts['gradient'])
                    lgv = self.ps.find_large_gradient_vars(gradient, threshold=large_gradient_threshold)
                    if len(lgv) > 0:
                        print "parameters with large gradient: "
                        print lgv

                if print_gradient_info:
                    gradient = gather(sts['gradient'])
                    gradient_magnitude = np.sqrt(np.sum(gradient ** 2))
                    print "|gradient| = %.3f" % gradient_magnitude

                if print_parameters:
                    pars = gather(self.ps.data)
                    pars_var = self.ps.split_gradient(pars)
                    print "parameters at iteration %d:" % iter
                    for name, value in pars_var.iteritems():
                        print "%10s: %s" % (name, repr(list(value)))

                if print_gradient:
                    gradient = gather(sts['gradient'])
                    gradient_var = self.ps.split_gradient(gradient)
                    print "gradient at iteration %d:" % iter
                    for name, value in gradient_var.iteritems():
                        print "%10s: %s" % (name, repr(list(value)))

                # if gradient is available anyway, check for NaNs and Infs
                if gradient is not None:
                    if not np.all(np.isfinite(gradient)):
                        his.should_terminate = True
                        his.termination_reason = 'inf_or_nan_gradient'
                        break

                if self.next_minibatch():
                    # iteration finished
                    iter += 1

                    # calculate losses
                    if iter % loss_record_interval == 0:
                        if self.record_loss(his, iter):
                            break

                # save checkpoint if necessary
                if checkpoint_handler is not None:
                    if checkpoint_handler.requested:
                        his.stop()
                        checkpoint_handler.save(data=gather(self.ps.data), his=his, iter=iter)
                    if his.should_save_checkpoint:
                        checkpoint_handler.save(data=gather(self.ps.data), his=his, iter=iter, explicit=True)
                        his.checkpoint_saved()

        self.after_training(his)

        # save results and plot loss
        if checkpoint_handler:
            his.stop()
            checkpoint_handler.save(data=gather(self.ps.data), his=his, iter=iter, explicit=True)
        his.finish()

        return his

    def after_training(self, his):
        """
        Called by generic_training after training is finished.
        For example this function can be used to add custom error measures to the training history.
        :param his: used training historz
        """
        pass