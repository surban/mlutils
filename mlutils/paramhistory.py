import time
import matplotlib.pyplot as plt
import numpy as np
import gnumpy as gp
import progress
from misc import get_key


class ParameterHistory(object):
    """Keeps track of parameter history, corresponding loses and optimization
    termination criteria."""

    def __init__(self, max_missed_val_improvements=200, show_progress=True,
                 desired_loss=None, min_improvement=0.00001, max_iters=None,
                 min_iters=None):
        self.max_missed_val_improvements = max_missed_val_improvements
        self.show_progress = show_progress
        self.desired_loss = desired_loss
        self.min_improvement = min_improvement
        self.max_iters = max_iters
        self.min_iters = min_iters

        self.best_val_loss = float('inf')
        self.best_tst_loss = float('inf')
        self.history = np.zeros((4, 0))
        self.last_val_improvement = 0
        self.should_terminate = False
        self.start_time = time.time()
        self.end_time = time.time()
        self.best_iter = None

        self.reset_best()

    def reset_best(self):
        self.best_val_loss = float('inf')
        self.last_val_improvement = 0
        self.should_terminate = False
        self.best_iter = None

    def add(self, iter, pars, trn_loss, val_loss, tst_loss):
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
        if (self.max_missed_val_improvements is not None and iter -
                self.last_val_improvement > self.max_missed_val_improvements):
            self.should_terminate = True
        if self.desired_loss is not None and val_loss <= self.desired_loss:
            self.should_terminate = True
        if self.min_iters is not None and iter < self.min_iters:
            self.should_terminate = False
        if self.max_iters is not None and iter >= self.max_iters:
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
            self.should_terminate = True

    def plot(self, final=True, logscale=True):
        self.end_time = time.time()

        if 'figsize' in dir(plt):
            plt.figsize(10, 5)
        # plt.clf()
        plt.hold(True)
        if logscale:
            plt.yscale('log')
            # plt.xscale('log')
        plt.plot(self.history[0], self.history[1], 'b')
        plt.plot(self.history[0], self.history[2], 'c')
        plt.plot(self.history[0], self.history[3], 'r')
        yl = plt.ylim()
        if self.best_iter is not None:
            plt.vlines(self.best_iter, yl[0], yl[1])
        plt.xlabel('iteration')
        plt.ylabel('loss')
        plt.legend(['training', 'validation', 'test'])

        if final and self.best_iter is not None:
            print "best iteration: %5d  best validation test loss: %9.5f  " \
                  "best test loss: %9.5f" % \
                  (self.best_iter, self.best_val_loss, self.best_tst_loss)
            print "training took %.2f s" % (self.end_time - self.start_time)

    @property
    def performed_iterations(self):
        return np.max(self.history[0])

    @property
    def converged(self):
        return self.best_val_loss <= self.desired_loss + self.min_improvement