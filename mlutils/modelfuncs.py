import numpy as np
from os.path import exists, join
from os import unlink
from mlutils import xp
from mlutils.config import optimizer_from_cfg, optimizers_from_cfg
from mlutils.dataset import Dataset
from mlutils.gpu import post, gather
from mlutils.parameterhistory import ParameterHistory
from mlutils.parameterlogger import ParameterLogger


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
        constants = self.ps.constants
        for name in dir(self.cfg):
            if name.startswith(prefix):
                varname = name[len(prefix):]
                value = getattr(self.cfg, name)
                print "Constant parameter: %015s = %s" % (varname, repr(value))
                constants[varname] = post(value)
        self.ps.constants = constants

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
        :type: mlutils.parameterset.ParameterSet
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
        print "Randomly initializing parameters with variance %f" % var
        self.ps.data[:] = post(np.random.normal(0, var, size=self.ps.data.shape))

        self.init_parameters_from_cfg()
        self.ps.restore_constants()

    def init_parameters_from_cfg(self):
        """
        Sets initial parameters in the ParameterSet from the configuration.
        """
        prefix = 'initval_'
        for name in dir(self.cfg):
            if name.startswith(prefix):
                varname = name[len(prefix):]
                value = getattr(self.cfg, name)
                print "Parameter initialization: %015s = %s" % (varname, repr(value))
                if isinstance(value, (float, int)):
                    self.ps[varname] = value
                else:
                    self.ps[varname] = post(value)

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
                         print_parameters=False, log_parameters=[], plot_logged_parameters=True,
                         print_logged_parameters=False, check_gradient_finite=False):
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
        :param log_parameters: list of parameter names (from self.ps) that should be logged every iteration
                               to file params.out
        :param plot_logged_parameters: if True, logged parameters are plotted to params.png
        :param print_logged_parameters: if True, logged parameters are also printed to standard output
        :param check_gradient_finite: if True, gradient is checked for infs and nans in every iteration
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
        if isinstance(self.cfg.optimizer, dict):
            def wrt_fprime_for_part(partition):
                wrt_for_part = self.ps.num_partition(partition)
                def fprime_for_part(pv_part):
                    # we assume that the optimizer updates the ParameterSet inplace and
                    # evaluates the gradient at the current values of the parameters
                    start, stop = self.ps.extents_of_partition(partition)
                    return grad_func(self.ps.num_data)[start : stop]
                return wrt_for_part, fprime_for_part
            opts_obj = optimizers_from_cfg(self.cfg, wrt_fprime_for_part, self.mb_loss)
            opts = {part: iter(opt_obj) for part, opt_obj in opts_obj.iteritems()}
            partioned_opt = True

            opt_parts = set(opts.keys())
            ps_parts = set(self.ps.partitions)
            if opt_parts != ps_parts:
                raise ValueError("optimizer config does not cover all ParameterSet partitions or vice versa: %s" %
                                 repr(opt_parts ^ ps_parts))
        else:
            opt = iter(optimizer_from_cfg(self.cfg, self.ps.data, self.mb_loss, grad_func))
            partioned_opt = False

        # initialize or restore checkpoint, if available
        if not checkpoint:
            itr = 0
            if initialize:
                self.init_parameters()

            his = ParameterHistory(cfg=self.cfg, state_dir=cfg_dir, max_iters=self.cfg.max_iters,
                                   max_missed_val_improvements=max_missed_val_improvements,
                                   desired_loss=desired_loss, iteration_gain=iteration_gain)
            logger = ParameterLogger(out_dir=cfg_dir, parameters=log_parameters,
                                     plot=plot_logged_parameters, print_stdout=print_logged_parameters)

            # Record initial loss and parameters
            self.record_loss(his, itr)
            logger.log(itr, self.ps)
        else:
            itr = checkpoint['iter']
            self.ps.data[:] = post(checkpoint['data'])

            his = checkpoint['his']
            his.state_dir = cfg_dir
            his.max_missed_val_improvements = max_missed_val_improvements
            his.desired_loss = desired_loss
            his.iteration_gain = iteration_gain

            # start and endtimes in his should have the same length, this is
            # not the case in explicit auto-save checkpoints, therefore set the
            # end to the saved
            if len(his.start_time) != len(his.end_time):
                his.end_time.append(checkpoint['save_time'])
            his.start()

            logger = checkpoint['logger']

        # reset termination criteria if requested
        second_chance_file = join(cfg_dir, "2nd_chance")
        if exists(second_chance_file):
            print "Resetting termination criteria because %s is present" % second_chance_file
            reset_termination_criteria = True
            unlink(second_chance_file)
        if reset_termination_criteria:
            his.should_terminate = False

        # do training
        self.ps.restore_constants()
        last_pars = xp.copy(self.ps.data)
        while not his.should_terminate:

            # call optimizer(s)
            if partioned_opt:
                for part, opt in opts.iteritems():
                    # print "optimizing %s" % part
                    opt.next()
            else:
                opt.next()

            # element change cap
            if 'step_element_cap' in dir(self.cfg) and self.cfg.step_element_cap is not None:
                d = self.ps.data - last_pars
                for par, lim in self.cfg.step_element_cap.iteritems():
                    start, stop = self.ps.extents_of_var(par)
                    dpar = d[start:stop]
                    # print "parameter diff for %s is %s (limit is %.4f)" % (par, repr(dpar), lim)
                    elems = xp.where(xp.abs(dpar) > lim)
                    dpar[elems] = xp.sign(dpar[elems]) * lim
                self.ps.data[:] = last_pars + d
                last_pars = xp.copy(self.ps.data)

            # parameter printout
            if print_parameters:
                pars = gather(self.ps.data)
                pars_var = self.ps.split(pars)
                print "parameters at iteration %d:" % itr
                for name, value in pars_var.iteritems():
                    print "%10s: %s" % (name, repr(list(value)))

            # obtain gradient if required for debugging operations
            if large_gradient_threshold > 0 or print_gradient_info or print_gradient:
                gradient = gather(grad_func(self.ps.num_data))
            else:
                gradient = None

            # check gradient for large elements
            if large_gradient_threshold > 0:
                lgv = self.ps.find_large_elements(gradient, threshold=large_gradient_threshold)
                if len(lgv) > 0:
                    print "parameters with large gradient: "
                    for (var, idx), value in lgv.itervalues():
                        print "                                %s[%d] = %.3f" % (var, idx, value)

            # gradient magnitude printout
            if print_gradient_info:
                gradient_magnitude = np.sqrt(np.sum(gradient ** 2))
                print "|gradient| = %.3f" % gradient_magnitude

            # gradient printout
            if print_gradient:
                gradient_var = self.ps.split(gradient)
                print "gradient at iteration %d:" % itr
                for name, value in gradient_var.iteritems():
                    print "%10s: %s" % (name, repr(list(value)))

            # check gradient for NaNs and Infs
            if check_gradient_finite or gradient is not None:
                if not np.all(np.isfinite(gradient)):
                    his.should_terminate = True
                    his.termination_reason = 'inf_or_nan_gradient'
                    break

            if self.next_minibatch():
                # iteration finished
                self.after_iteration(his, itr)

                itr += 1

                # log parameters
                logger.log(itr, self.ps)

                # calculate losses
                if itr % loss_record_interval == 0:
                    self.record_loss(his, itr)

            # save checkpoint if necessary
            if checkpoint_handler is not None:
                if checkpoint_handler.requested:
                    his.stop()
                    checkpoint_handler.save(data=gather(self.ps.data), his=his, iter=itr, logger=logger)
                if his.should_save_checkpoint:
                    checkpoint_handler.save(data=gather(self.ps.data), his=his, iter=itr, logger=logger,
                                            explicit=True)
                    his.checkpoint_saved()

        # training finished
        self.after_training(his)

        # save results and plot loss
        if checkpoint_handler:
            his.stop()
            checkpoint_handler.save(data=gather(self.ps.data), his=his, iter=itr, logger=logger, explicit=True)
        his.finish()
        logger.plot()

        return his

    def after_iteration(self, his, iteration):
        """
        Called by generic_training after one training iteration has been performed.
        :param his: used training history
        :param iteration: iteration number of just finished training iteration
        """
        pass

    def after_training(self, his):
        """
        Called by generic_training after training is finished.
        For example this function can be used to add custom error measures to the training history.
        :param his: used training history
        """
        pass