from glob import glob
import json
import time
from types import ModuleType
import matplotlib.pyplot as plt
import numpy as np
import gnumpy as gp
from os.path import join, split, exists
import sys
from mlutils.gpu import gather, post
import progress
from misc import get_key


class ParameterHistory(object):
    """Keeps track of parameter history, corresponding loses and optimization
    termination criteria."""

    def __init__(self, cfg=None,
                 show_progress=True, state_dir=None,
                 desired_loss=None,
                 max_iters=None, min_iters=None,
                 max_missed_val_improvements=200, min_improvement=1e-7):
        """
        Creates a ParameterHistory object that tracks loss, best parameters and termination criteria during training.
        Training is performed until there is no improvement of validation loss for
        max_missed_val_improvements iterations.
        Call add for each performed iteration.
        Exit the training loop if should_terminate becomes True.
        Call finish once the training loop has been exited.
        :param cfg: configuration dictionary or module
        :param show_progress: if True, loss is printed during training
        :param state_dir: directory where results and plots are saved
        :param desired_loss: loss that should lead to immediate termination of training when reached
        :param max_iters: maximum number of iterations
        :param min_iters: minimum number of iterations before improvement checking is done
        :param max_missed_val_improvements: maximum number of iterations without improvement of validation loss
        before training is terminated
        :param min_improvement: minimum change in loss to count as improvement
        """
        if cfg is None:
            self.cfg = {}
        elif isinstance(cfg, ModuleType):
            self.cfg = cfg_module_to_dict(cfg)
        else:
            self.cfg = dict(cfg)
        self.show_progress = show_progress
        self.state_dir = state_dir

        self.desired_loss = desired_loss
        self.min_iters = min_iters
        self.max_iters = max_iters
        self.max_missed_val_improvements = max_missed_val_improvements
        self.min_improvement = min_improvement

        self.best_val_loss = float('inf')
        self.best_tst_loss = float('inf')
        self.history = np.zeros((4, 0))
        self.last_val_improvement = 0
        self.should_terminate = False
        self.start_time = time.time()
        self.end_time = time.time()
        self.best_iter = None
        self.best_pars = None
        self.termination_reason = ''

        self.reset_best()

    def reset_best(self):
        """
        Resets the best iteration statistics.
        """
        self.best_val_loss = float('inf')
        self.last_val_improvement = 0
        self.should_terminate = False
        self.best_iter = None
        self.best_pars = None

    def add(self, iter, pars, trn_loss, val_loss, tst_loss):
        """
        Adds an iteration.
        :param iter: iteration number
        :param pars: parameters used in this iteration
        :param trn_loss: training set loss
        :param val_loss: validation set loss
        :param tst_loss: test set loss
        """
        iter = int(iter)
        trn_loss = float(trn_loss)
        val_loss = float(val_loss)
        tst_loss = float(tst_loss)

        # keep track of best results so far
        if val_loss < self.best_val_loss - self.min_improvement:
            self.best_iter = iter
            self.best_val_loss = val_loss
            self.best_tst_loss = tst_loss
            if isinstance(pars, gp.garray):
                self.best_pars = gp.garray(pars, copy=True)
            else:
                self.best_pars = np.copy(pars)
            self.last_val_improvement = iter

        # termination criteria
        if (self.max_missed_val_improvements is not None and
                iter - self.last_val_improvement > self.max_missed_val_improvements):
            self.termination_reason = 'no_improvement'
            self.should_terminate = True
        if self.desired_loss is not None and val_loss <= self.desired_loss:
            self.termination_reason = 'desired_loss_reached'
            self.should_terminate = True
        if self.min_iters is not None and iter < self.min_iters:
            self.termination_reason = ''
            self.should_terminate = False
        if self.max_iters is not None and iter >= self.max_iters:
            self.termination_reason = 'max_iters_reached'
            self.should_terminate = True

        # store current losses
        self.history = np.hstack((self.history, [[iter],
                                                 [trn_loss],
                                                 [val_loss],
                                                 [tst_loss]]))

        # display progress
        if self.show_progress:
            caption = "training: %9.5f  validation: %9.5f (best: %9.5f)  " \
                      "test: %9.5f" % (trn_loss, val_loss, self.best_val_loss,
                                       tst_loss)
            progress.status(iter, caption=caption)

        # termination by user
        if get_key() == "q":
            print
            print "Termination by user."
            self.termination_reason = 'user'
            self.should_terminate = True

    def plot(self, logscale=True):
        """
        Plots the loss history.
        :param logscale: if True, logarithmic scale is used
        """
        if 'figsize' in dir(plt):
            plt.figsize(10, 5)
        plt.hold(True)
        try:
            if logscale:
                plt.yscale('log')
                # plt.xscale('log')
            plt.plot(self.history[0], self.history[1], 'b')
            plt.plot(self.history[0], self.history[2], 'c')
            plt.plot(self.history[0], self.history[3], 'r')
        except ValueError:
            # catches: ValueError: Data has no positive values, and therefore can not be log-scaled.
            # when no data is present or we only have NaNs
            pass
        yl = plt.ylim()
        if self.best_iter is not None:
            plt.vlines(self.best_iter, yl[0], yl[1])
        plt.xlabel('iteration')
        plt.ylabel('loss')
        plt.legend(['training', 'validation', 'test'])

    @property
    def performed_iterations(self):
        """Number of performed iterations."""
        return np.max(self.history[0])

    @property
    def converged(self):
        """True if the desired loss has been achieved."""
        return self.best_val_loss <= self.desired_loss + self.min_improvement

    @property
    def training_time(self):
        """The time training took in seconds."""
        return self.end_time - self.start_time

    @staticmethod
    def _get_result_filenames(cfg_dir):
        history_filename = join(cfg_dir, "history.npz")
        results_filename = join(cfg_dir, "results.json")
        best_pars_filename = join(cfg_dir, "best_pars.npz")
        return history_filename, results_filename, best_pars_filename

    def save(self):
        """
        Saves the parameter history and best results to disk in the state directory.
        """
        history_filename, results_filename, best_pars_filename = self._get_result_filenames(self.state_dir)
        np.savez_compressed(history_filename, history=self.history)
        np.savez_compressed(best_pars_filename, best_pars=gather(self.best_pars))
        with open(results_filename, 'wb') as results_file:
            json.dump({'best_iter': self.best_iter,
                       'best_val_loss': self.best_val_loss,
                       'best_tst_loss': self.best_tst_loss,
                       'training_time': self.training_time,
                       'start_time': self.start_time,
                       'end_time': self.end_time,
                       'termination_reason': self.termination_reason,
                       'cfg': self.cfg},
                      results_file, indent=4)

    @classmethod
    def load(cls, state_dir):
        """
        Loads parameters history and best results from the state directory.
        :param state_dir: directory to load from
        :return: a ParameterHistory object loaded from the state directory
        """
        history_filename, results_filename, best_pars_filename = cls._get_result_filenames(state_dir)
        self = cls()
        self.history = np.load(history_filename)['history']
        self.best_pars = post(np.load(best_pars_filename)['best_pars'])
        with open(results_filename, 'rb') as results_file:
            data = json.load(results_file)
            self.best_iter = data['best_iter']
            self.best_val_loss = data['best_val_loss']
            self.best_tst_loss = data['best_tst_loss']
            self.start_time = data['start_time']
            self.end_time = data['end_time']
            self.termination_reason = data['termination_reason']
            if 'cfg' in data:
                self.cfg = data['cfg']
            else:
                self.cfg = {}
        return self

    @classmethod
    def load_resultset(cls, base_dir):
        """
        Loads all results from the subdirectories of base_dir.
        :param base_dir:
        :return:
        """
        his = {}
        for results_filename in glob(join(base_dir, "*", "results.json")):
            results_dir, _ = split(results_filename)
            _, cfg_name = split(results_dir)
            his[cfg_name] = cls.load(results_dir)
        return his

    def finish(self):
        """
        Marks training as finished. Should be called right after exiting the training loop.
        Prints statistics, saves results to disk in the state directory and plots the loss curve.
        """
        self.end_time = time.time()

        # print statistics
        if self.best_iter is not None:
            print "Best iteration %5d with validation loss %9.5f and test loss %9.5f" % \
                  (self.best_iter, self.best_val_loss, self.best_tst_loss)
            print "Training took %.2f s and was terminated because %s." % (self.end_time - self.start_time,
                                                                           self.termination_reason)

        # save results
        self.save()

        # plot loss curve
        plt.figure()
        self.plot()
        plt.savefig(join(self.state_dir, "loss.pdf"))

        # change directory icon to indicate that task is finished
        if sys.platform == 'win32':
            try:
                import win32api, win32con
                desktopfile = join(self.state_dir, "desktop.ini")
                if exists(desktopfile):
                    win32api.SetFileAttributes(desktopfile, win32con.FILE_ATTRIBUTE_NORMAL)
                with open(desktopfile, 'w') as df:
                    df.write("[.ShellClassInfo]\n")
                    df.write("IconResource=C:\\WINDOWS\\system32\\SHELL32.dll,144\n")
                win32api.SetFileAttributes(desktopfile, win32con.FILE_ATTRIBUTE_SYSTEM | win32con.FILE_ATTRIBUTE_HIDDEN)
                win32api.SetFileAttributes(self.state_dir, win32con.FILE_ATTRIBUTE_READONLY)
            except Exception, e:
                print "Could not write desktop.ini: ", e

def cfg_module_to_dict(mod):
    """
    Constructs a dictionary from all variables in a module.
    :param mod: the module
    :return: dictionary of all variables in the module
    """
    cfg = {}
    for name in dir(mod):
        val = getattr(mod, name)
        if isinstance(val, (int, float, long, basestring, bool)) and not name.startswith('__'):
            cfg[name] = val
    return cfg
