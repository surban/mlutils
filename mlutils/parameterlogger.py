from os.path import join
from mlutils.gpu import gather
import matplotlib.pyplot as plt
import numpy as np

class ParameterLogger(object):
    """
    Parameter logger.
    """

    def __init__(self, out_dir, parameters=[], plot=True, print_stdout=False):
        """
        Parameter logger.
        :param out_dir: directory to log to
        :param parameters: list of parameter names (from the ParameterSet) that should be logged
                               to file params.out
        :param plot: if True, logged parameters are plotted to params.pdf
        """
        self._out_dir = out_dir
        self._parameters = parameters
        self._plot = plot
        self._print_stdout = print_stdout

        self._history = {}
        self._itrs = []

        self.init()

    @property
    def log_filename(self):
        return join(self._out_dir, "params.out")

    @property
    def plot_filename(self):
        return join(self._out_dir, "params.pdf")

    @property
    def active(self):
        return len(self._parameters) > 0

    def init(self):
        """
        Clears the parameter history.
        """
        if not self.active:
            return

        # clear log file
        with open(self.log_filename, 'w') as _:
            pass

        # clear plot history
        if self._plot:
            self._itrs = []
            self._history = {par: [] for par in self._parameters}

    def log(self, itr, ps):
        """
        Logs parameters from the specified ParameterSet.
        :param itr: iteration number
        :param ps: ParameterSet to obtain parameter values from
        """
        if not self.active:
            return

        # generate log line
        line = "%5d:  " % itr
        for par in self._parameters:
            val = gather(ps[par])
            if val.size == 1:
                val = "%.4f" % val.flat[0]
            else:
                val = "\n" + repr(val)
            line += "%s=%s  " % (par, val)
        line += "\n"

        # log to file
        with open(self.log_filename, 'a') as logfile:
            logfile.write(line)

        # log to stdout
        if self._print_stdout:
            print line,

        # log to buffer for plotting
        if self._plot:
            self._itrs.append(itr)
            for par in self._parameters:
                self._history[par].append(gather(ps[par]).flatten())

    def plot(self):
        """
        Plots the logged parameters.
        """
        if not self.active:
            return

        if self._plot:
            plt.figure()
            n_pars = len(self._parameters)
            for idx, par in enumerate(self._parameters):
                plt.subplot(n_pars, 1, idx + 1)
                data = np.asarray(self._history[par])
                plt.plot(self._itrs, self._history[par])
                plt.ylabel(par)
                if idx == n_pars - 1:
                    plt.xlabel("iteration")
                else:
                    plt.xticks([])
            plt.tight_layout()
            plt.savefig(self.plot_filename)
            plt.close()


