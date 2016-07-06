import numpy as np
import matplotlib.pyplot as plt
from os.path import exists, join
from os import unlink
from mlutils import xp
from mlutils.config import optimizer_from_cfg, optimizers_from_cfg, load_cfg
from mlutils.dataset import Dataset
from mlutils.gpu import post, gather
from mlutils.parameterhistory import ParameterHistory
from mlutils.parameterlogger import ParameterLogger
from mlutils.misc import n_steps_to_valid, valid_to_n_steps
from mlutils.plot import plot_weight_histograms
from mlutils.preprocess import pca_white, for_step_data, pca_white_inverse
from mlutils.gitlogger import git_log


class ModelFuncs(object):
    """Base class for callable functions of a Theano model."""

    def __init__(self, model=None, cfg=None, dataset=None):
        if cfg.preprocess_pca is not None and cfg.preprocess_pca != cfg.n_units[0]:
            raise ValueError("number of PCA components must match input unit count")
        if cfg.preprocess_pca is not None and cfg.loss == 'cross_entropy':
            raise ValueError("PCA whitening does not work with cross entropy loss")

        self.cfg = cfg

        if model is not None:
            self.model = model
        else:
            self.model = self.create_model(cfg)

        if dataset is not None:
            self.dataset = dataset
        else:
            fractions = cfg.dataset_fractions if 'dataset_fractions' in dir(cfg) else [0.8, 0.1, 0.1]
            data = self.build_dataset(self.cfg, getattr(self.cfg, 'dataset', None))
            if data is None:
                data = self.cfg.dataset
            self.dataset = Dataset(data, 
                                   minibatch_size=self.cfg.minibatch_size,
                                   fractions=fractions,
                                   preprocessing_function=self.preprocess_dataset)

        self._minibatch_idx = 0
        self._minibatch = None
        self.minibatch_idx = 0

        self.set_constant_parameters_from_cfg()

    def build_dataset(self, cfg, path):
        """
        Builds the dataset dictionary.
        :param cfg: configuration
        :param path: dataset location as specified in config
        :return: dataset dictionary
        """
        return None

    def preprocess_dataset(self, ds):
        """
        Preprocesses the dataset.
        :param ds: dataset
        :return: preprocessed dataset
        """
        if self.cfg.dataset_input != 'input':
            ds['input'] = ds[self.cfg.dataset_input]
            del ds[self.cfg.dataset_input]
        if self.cfg.dataset_target != 'target':
            ds['target'] = ds[self.cfg.dataset_target]
            del ds[self.cfg.dataset_target]

        if ds['input'].ndim == 3:
            # ensure that n_steps and valid are both in the dataset
            if 'n_steps' in ds and 'valid' not in ds:
                ds['valid'] = n_steps_to_valid(ds['n_steps'], ds['input'].shape[1])
            elif 'valid' in ds and 'n_steps' not in ds:
                ds['n_steps'] = valid_to_n_steps(ds['valid'])

        if self.cfg.dataset_samples is not None:
            ds['input'] = ds['input'][..., 0:self.cfg.dataset_samples]
            ds['target'] = ds['target'][..., 0:self.cfg.dataset_samples]
            print "Using only %d samples from dataset" % ds['input'].shape[-1]

        if self.cfg.no_negative_data:
            minval = np.min(ds['input'])
            if minval < 0:
                print "Adding %.3f to dataset inputs to ensure positive values." % (-minval)
                ds['input'] -= minval
            else:
                print "Dataset inputs are already positive."

        if self.cfg.preprocess_pca is not None:
            ds['orig_input'] = np.copy(ds['input'])
            if ds['input'].ndim == 2:
                res = pca_white(ds['input'], n_components=self.cfg.preprocess_pca, return_axes=True)
            elif ds['input'].ndim == 3:
                res = for_step_data(pca_white)(ds['n_steps'], ds['input'], n_components=self.cfg.preprocess_pca, return_axes=True)
            else:
                raise ValueError("unrecognized dimensionality of  input variable")
            ds['input'], ds['meta_pca_vars'], ds['meta_pca_axes'], ds['meta_pca_means'] = res                
            print "Keeping %d principal components (PCA) with variances:" % self.cfg.preprocess_pca
            print ds['meta_pca_vars']
            np.savez_compressed(join(self.cfg.out_dir, "pca.npz"),
                                pca_vars=ds['meta_pca_vars'],
                                pca_axes=ds['meta_pca_axes'],
                                pca_means=ds['meta_pca_means'])

        if self.cfg.use_training_as_validation:
            ds['meta_use_training_as_validation'] = self.cfg.use_training_as_validation

        return ds

    def perform_pca(self, data):
        """
        Performs the same PCA whitening as done during preprocessing.
        :param data: data[feature, smpl] or data[feature, step, smpl]
        :return: whitened[comp, smpl] or whitened[comp, step, smpl]
        """
        if self.cfg.preprocess_pca is not None:
            if data.ndim == 2:
                return pca_white(data,
                                 variances=self.dataset.meta_pca_vars,
                                 axes=self.dataset.meta_pca_axes,
                                 means=self.dataset.meta_pca_means)
            elif data.ndim == 3:
                n_steps = np.full((data.shape[2],), data.shape[1], dtype=int)
                return for_step_data(pca_white)(n_steps, data,
                                                variances=self.dataset.meta_pca_vars,
                                                axes=self.dataset.meta_pca_axes,
                                                means=self.dataset.meta_pca_means)
        else:
            return data

    def invert_pca(self, whitened):
        """
        Inverts the PCA whitening done during preprocessing.
        :param whitened: whitened[comp, smpl] or whitened[comp, step, smpl]
        :returns: data[feature, smpl] or data[feature, step, smpl]
        """
        if self.cfg.preprocess_pca is not None:
            if whitened.ndim == 2:
                return pca_white_inverse(whitened,
                                         self.dataset.meta_pca_vars,
                                         self.dataset.meta_pca_axes,
                                         self.dataset.meta_pca_means)
            elif whitened.ndim == 3:
                n_steps = np.full((whitened.shape[2],), whitened.shape[1], dtype=int)
                return for_step_data(pca_white_inverse)(n_steps, whitened,
                                                        self.dataset.meta_pca_vars,
                                                        self.dataset.meta_pca_axes,
                                                        self.dataset.meta_pca_means)
        else:
            return whitened
            
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
        if self.model is not None:
            return self.model.ps
        else:
            return self._ps

    def init_parameters(self, var=None, seed=None):
        """Initializes the parameteres of the ParameterSet using a zero-mean normal distribution.
        :param var: Variance of the normal distribution.
                    If None, then the 'initialization_variance' parameter from the configuration is used.
        :param seed: random seed for initialization
        """
        if var is None:
            if 'initialization_variance' in dir(self.cfg):
                var = self.cfg.initialization_variance
            elif 'initvar' in dir(self.cfg):
                var = self.cfg.initvar
            else:
                var = 0.01
        if seed is None:
            if 'initialization_seed' in dir(self.cfg):
                seed = self.cfg.initialization_seed
            else:
                seed = 1
        print "Random parameter initialization: ALL ~ N(0, %.3f) with seed %d" % (var, seed)
        np.random.seed(seed)
        self.ps.data[:] = post(np.random.normal(0, var, size=self.ps.data.shape))

        if self.cfg.positive_weights_init:
            print "Ensuring positive initialization weights."
            self.ps.data[:] = xp.abs(self.ps.data)

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

        prefix = 'initvar_'
        for name in dir(self.cfg):
            if name.startswith(prefix):
                varname = name[len(prefix):]
                variance = getattr(self.cfg, name)
                meanname = 'initmean_' + varname
                mean = getattr(self.cfg, meanname) if meanname in dir(self.cfg) else 0.0
                print "Random parameter initialization: %015s ~ N(%.3f, %.3f)" % (varname, mean, variance)
                self.ps[varname] = post(np.random.normal(mean, variance, size=self.ps[varname].shape))

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
                         max_missed_val_improvements=200, iteration_gain=1.25, min_improvement=1e-7,
                         reset_termination_criteria=False, desired_loss=None, initialize=True,
                         large_gradient_threshold=0.0, print_gradient_info=False, print_gradient=False,
                         print_parameters=False, log_parameters=[], plot_logged_parameters=True,
                         print_logged_parameters=False, check_gradient_finite=False, log_modules=None):
        """
        Generic training procedure.
        :param cfg_dir: configuration directory
        :param checkpoint: checkpoint
        :param checkpoint_handler: checkpoint handler
        :param loss_record_interval: number of iterations between calculating and recording losses
        :param max_missed_val_improvements: maximum iterations without improvement of loss before training ist stopped
        :param iteration_gain: If not set to 0, then training is performed up to iteration
                               (iteration_gain * iteration_of_last_improvement).
        :param min_improvement: minimum loss change to count as improvement
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
        :param log_modules: list of python modules for which version or latest git commit should be logged
        :return: ParameterHistory object of training
        """

        max_iters = self.cfg.max_iters if 'max_iters' in dir(self.cfg) else None

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

        # initialize or restore checkpoint, if available
        if not checkpoint:
            itr = 0
            if initialize:
                self.init_parameters()

            his = ParameterHistory(cfg=self.cfg, state_dir=cfg_dir, max_iters=max_iters,
                                   max_missed_val_improvements=max_missed_val_improvements,
                                   min_improvement=min_improvement,
                                   desired_loss=desired_loss, iteration_gain=iteration_gain)
            logger = ParameterLogger(out_dir=cfg_dir, parameters=log_parameters,
                                     plot=plot_logged_parameters, print_stdout=print_logged_parameters)
            git_log(modules=log_modules, log_dir=cfg_dir)

            # Record initial loss and parameters
            self.record_loss(his, itr)
            logger.log(itr, self.ps)
        else:
            itr = checkpoint['iter']
            self.ps.data[:] = post(checkpoint['data'])
            if 'optimizer_step_rate' in checkpoint:
                self.cfg.optimizer_step_rate = checkpoint['optimizer_step_rate']

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
            git_log(modules=log_modules, log_dir=cfg_dir, check=True)

        # reset termination criteria if requested
        second_chance_file = join(cfg_dir, "2nd_chance")
        if exists(second_chance_file):
            print "Resetting termination criteria because %s is present" % second_chance_file
            reset_termination_criteria = True
            unlink(second_chance_file)
        if self.cfg.continue_training:
            print "Resetting termination criteria because --continue flag was specified"
            reset_termination_criteria = True
        if reset_termination_criteria:
            his.reset_best()

        if 'step_element_cap' in dir(self.cfg):
            step_element_cap_orig = self.cfg.step_element_cap
        step_element_cap_decrease_iteration = None

        restart = True
        while restart and (max_iters is None or max_iters > 0):
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
                    if isinstance(self.cfg.step_element_cap, dict):
                        for par, lim in self.cfg.step_element_cap.iteritems():
                            start, stop = self.ps.extents_of_var(par)
                            dpar = d[start:stop]    # dpar is a subview of d
                            # print "parameter diff for %s is %s (limit is %.4f)" % (par, repr(dpar), lim)
                            elems = xp.where(xp.abs(dpar) > lim)
                            dpar[elems] = xp.sign(dpar[elems]) * lim
                    elif isinstance(self.cfg.step_element_cap, (float, int)):
                        lim = float(self.cfg.step_element_cap)
                        elems = xp.where(xp.abs(d) > lim)
                        d[elems] = xp.sign(d[elems]) * lim
                    else:
                        raise TypeError("cfg.step_element_cap must either be a dict or a float")
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

                    if step_element_cap_decrease_iteration is not None:
                        if 'step_element_cap_restore_iterations' in dir(self.cfg):
                            restore_itrs = self.cfg.step_element_cap_restore_iterations
                        else:
                            restore_itrs = 100
                        if itr >= step_element_cap_decrease_iteration + restore_itrs:
                            self.cfg.step_element_cap = step_element_cap_orig
                            print "Restored step element cap to %g" % self.cfg.step_element_cap
                            step_element_cap_decrease_iteration = None

                # save checkpoint if necessary
                if checkpoint_handler is not None:
                    if checkpoint_handler.requested:
                        his.stop()
                        checkpoint_handler.save(data=gather(self.ps.data), his=his, iter=itr, logger=logger,
                                                optimizer_step_rate=self.cfg.optimizer_step_rate)
                    if his.should_save_checkpoint:
                        checkpoint_handler.save(data=gather(self.ps.data), his=his, iter=itr, logger=logger,
                                                optimizer_step_rate=self.cfg.optimizer_step_rate,
                                                explicit=True)
                        his.checkpoint_saved()

            # restore best parametes
            self.ps.data[:] = his.best_pars

            # check for retry conditions
            restart = False

            # temporarily reduce step element cap to move over regions with very large gradient
            if (his.should_terminate and his.termination_reason == 'nan_or_inf_loss' and
                    'step_element_cap' in dir(self.cfg) and 'step_element_cap_min' in dir(self.cfg) and
                    self.cfg.step_element_cap >= self.cfg.step_element_cap_min):
                self.cfg.step_element_cap /= 10.
                step_element_cap_decrease_iteration = itr
                print "Reduced step element cap to %g" % self.cfg.step_element_cap
                his.should_terminate = False
                restart = True

            # advance learning rate schedule
            if (his.should_terminate and
                    his.termination_reason in ['no_improvement', 'nan_or_inf_loss', 'user_learning_rate_decrease'] and
                    'optimizer_step_rate_min' in dir(self.cfg) and
                    self.cfg.optimizer_step_rate / 10. >= self.cfg.optimizer_step_rate_min):
                self.cfg.optimizer_step_rate /= 10.
                print "Decaying optimizer step rate to %g" % self.cfg.optimizer_step_rate
                his.should_terminate = False
                his.last_val_improvement = itr
                restart = True

        # training finished
        self.after_training(his)

        # save results and plot loss
        if checkpoint_handler:
            his.stop()
            checkpoint_handler.save(data=gather(self.ps.data), his=his, iter=itr, logger=logger,
                                    optimizer_step_rate=self.cfg.optimizer_step_rate,
                                    explicit=True)
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
        
    @classmethod
    def default_cfg(cls):
        return {'func_class': None,
                'dataset_samples': None,
                'dataset_input': 'input',
                'dataset_target': 'target',
                'max_missed_val_improvements': 100,
                'iteration_gain': 0,
                'preprocess_pca': None,
                'use_training_as_validation': False,
                'no_negative_data': False,
                'positive_weights_init': False,
                'minibatch_size': 200
               }.copy()

    @classmethod
    def train_from_cfg(cls, with_predictions=True):
        """
        Creates and trains model functions using configuration specified at command line.
        """
        # create model
        cfg, cfg_dir, cph, cp = load_cfg(defaults=cls.default_cfg(), with_checkpoint=True)
        if cfg.func_class is not None:
            funcs = cfg.func_class(cfg=cfg)
        else:
            funcs = cls(cfg=cfg)

        # train
        his = funcs.generic_training(cfg_dir, cp, cph,
                                     max_missed_val_improvements=cfg.max_missed_val_improvements,
                                     iteration_gain=cfg.iteration_gain)

        # set history field
        setattr(funcs, 'his', his)

        if with_predictions:
            # plot weight histogram
            plt.figure(figsize=(14, 14))
            plot_weight_histograms(funcs.ps, funcs.ps.all_vars())
            plt.savefig(join(cfg_dir, "weights.pdf"))
            plt.close()

            # obtain predictions
            trn_inp = gather(funcs.dataset.trn.input)
            trn_tgt = gather(funcs.dataset.trn.target)
            trn_pred = gather(funcs.predict(funcs.ps.data, funcs.dataset.trn))
            funcs.show_results('trn', funcs.dataset.trn.gather(), trn_inp, trn_tgt, trn_pred)

            val_inp = gather(funcs.dataset.val.input)
            val_tgt = gather(funcs.dataset.val.target)
            val_pred = gather(funcs.predict(funcs.ps.data, funcs.dataset.tst))
            funcs.show_results('val', funcs.dataset.val.gather(), val_inp, val_tgt, val_pred)

            tst_inp = gather(funcs.dataset.tst.input)
            tst_tgt = gather(funcs.dataset.tst.target)
            tst_pred = gather(funcs.predict(funcs.ps.data, funcs.dataset.tst))
            funcs.show_results('tst', funcs.dataset.tst.gather(), tst_inp, tst_tgt, tst_pred)
        else:
            trn_inp = None
            trn_tgt = None
            trn_pred = None
            val_inp = None
            val_tgt = None
            val_pred = None
            tst_inp = None
            tst_tgt = None
            tst_pred = None

        cph.release()
        return dict(cfg=cfg, cfg_dir=cfg_dir, funcs=funcs, his=his,
                    trn_inp=trn_inp, trn_tgt=trn_tgt, trn_pred=trn_pred,
                    val_inp=val_inp, val_tgt=val_tgt, val_pred=val_pred,
                    tst_inp=tst_inp, tst_tgt=tst_tgt, tst_pred=tst_pred)

    def show_results(self, partition, ds, inp, tgt, pred):
        """
        Should be overriden to show prediction results, for example plot and save to disk.
        :param partition: dataset partition, is 'trn', 'val' or 'tst'
        :param ds: corresponding dataset
        :param inp: inputs[smpl, ...]
        :param tgt: targets[smpl, ...]
        :param pred: predictions[smpl, ...]
        """
        pass

    def create_model(self, cfg):
        """
        Creates the employed model.
        :param cfg: configuration
        :return: the model
        """
        return cfg.model

