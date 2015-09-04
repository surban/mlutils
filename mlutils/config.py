from socket import gethostname
import ctypes
from inspect import getargspec
import os
import sys
import glob
import imp
import pickle
import signal
import itertools
import __main__ as main
from os.path import isfile
import climin


def multiglob(*patterns):
    return itertools.chain.from_iterable(
        glob.glob(pattern) for pattern in patterns)


def base_dir():
    """Finds the path to the base directory of the main module.
     The base directory is the first parent directory of the main module or,
     if no main module is loaded, the current directory, that contains a 'cfgs' subdirectory.
    :returns: path to the base directory"""
    try:
        p = os.path.dirname(main.__file__)
    except AttributeError:
        p = ""
    while not os.path.isdir(os.path.join(p, "cfgs")):
        p = os.path.join(p, "..")
        if len(p) > 500:
            raise RuntimeError("Cannot find base directory (contains a 'cfgs' subdirectory)")
    return p


def cfgs_dir():
    """Finds the path to the configuration directory.
    :returns: path to the configuration directory"""
    return os.path.join(base_dir(), "cfgs")


def load_cfg(config_name=None, prepend="", clean_outputs=False, with_checkpoint=False, defaults={}):
    """Reads the configuration file cfg.py from the configuration directory
    specified as the first parameter on the command line.
    Returns a tuple consisting of the configuration module and the plot directory.
    :param clean_outputs: plot and data files are removed from the config directory
    :param prepend: Prepend subdirectory for config file. Use %SCRIPTNAME% to insert name of running script.
    :param with_checkpoint: enables checkpoint support
    :param defaults: default values for non-specified configuration variables
    :returns: if with_checkpoint == True:  (cfg module, cfg directory, checkpoint handler, checkpoint)
              if with_checkpoint == False: (cfg module, cfg directory)
    """
    if config_name is None:
        if len(sys.argv) < 2:
            if with_checkpoint:
                print "Usage: %s <config> [restart]" % sys.argv[0]
            else:
                print "Usage: %s <config>" % sys.argv[0]
            sys.exit(1)
        else:
            config_name = sys.argv[1]

    print "Host: %s" % gethostname()

    try:
        scriptname, scriptext = os.path.splitext(os.path.basename(main.__file__))
        print "Script: %s" % (scriptname + scriptext)
    except AttributeError:
        scriptname = "unknown"

    prepend = prepend.replace("%SCRIPTNAME%", scriptname)
    cfgdir = os.path.join(cfgs_dir(), prepend, config_name)
    cfgname = os.path.join(cfgdir, 'cfg.py')
    if not os.path.exists(cfgname):
        raise RuntimeError("Config: %s not found" % cfgname)
    print "Config: %s" % cfgname
    sys.dont_write_bytecode = True
    cfg = imp.load_source('cfg', cfgname)

    # load checkpoint if requested
    checkpoint = None
    if with_checkpoint:
        cp_handler = CheckpointHandler(cfgdir)
        if (cp_handler.exists and not (len(sys.argv) >= 3 and sys.argv[2] == "restart")):
            checkpoint = cp_handler.load()
        else:
            print "Checkpoint: none"
            cp_handler.remove()

    # set defaults
    for k, v in defaults.iteritems():
        if k not in dir(cfg):
            setattr(cfg, k, v)

    # clean configuration directory
    if clean_outputs and checkpoint is None:
        curdir = os.path.abspath(os.curdir)
        os.chdir(cfgdir)
        for file in multiglob('*.png', '*.pdf', '*.npz'):
            os.remove(file)
        os.chdir(curdir)

    if with_checkpoint:
        return cfg, cfgdir, cp_handler, checkpoint
    else:
        return cfg, cfgdir


def optimizer_from_cfg(cfg, wrt, f, fprime):
    # get specified optimizer and its constructor
    class_name = 'climin.' + cfg.optimizer
    class_obj = eval(class_name)
    init_func = eval(class_name + '.__init__')

    # build constructor arguments
    args = getargspec(init_func).args
    kwargs = {}
    for arg in args[1:]:
        if arg == 'wrt':
            kwargs['wrt'] = wrt
        elif arg == 'f':
            if f is None:
                raise ValueError("optimizer requires f, but it was not specified")
            kwargs['f'] = f
        elif arg == 'fprime':
            kwargs['fprime'] = fprime
        else:
            cfg_arg = 'optimizer_' + arg
            if cfg_arg in dir(cfg):
                kwargs[arg] = getattr(cfg, cfg_arg)

    return class_obj(**kwargs)


class CheckpointHandler(object):
    def __init__(self, directory, filename="checkpoint.dat"):
        self._path = os.path.join(directory, filename)
        self._directory = directory
        self._requested = False

        if sys.platform == 'win32':
            # load Fortran DLLs before setting our own console control handler
            # because they replace it with their own handler
            basepath = imp.find_module('numpy')[1]
            ctypes.CDLL(os.path.join(basepath, 'core', 'libmmd.dll'))
            ctypes.CDLL(os.path.join(basepath, 'core', 'libifcoremd.dll'))

            # install win32 console control event handler
            import win32api
            win32api.SetConsoleCtrlHandler(self._console_ctrl_handler, 1)
        else:
            # install unix signal handlers
            signal.signal(signal.SIGINT, self._signal_handler)
            signal.signal(signal.SIGBREAK, self._signal_handler)

    def _signal_handler(self, signum, frame):
        print "Checkpoint: requested by termination signal"
        self._requested = True

    def _console_ctrl_handler(self, ctrl_type):
        if ctrl_type in [0, 1]:  # Ctrl-C or Ctrl-Break
            print "Checkpoint: requested by termination signal"
            self._requested = True
            return 1  # don't chain to the next handler
        return 0  # chain to the next handler

    @staticmethod
    def _replace_file(src, dest):
        if sys.platform == 'win32':
            if os.path.exists(dest):
                os.remove(dest)
            assert not os.path.exists(dest), "%s still exists after deleting it" % dest
        os.rename(src, dest)

    @property
    def requested(self):
        return self._requested

    def save(self, **kwargs):
        explicit = False
        if 'explicit' in kwargs:
            explicit = kwargs['explicit']

        if self._requested:
            print "Checkpoint: saving %s" % self._path
        if self._requested or explicit:
            with open(self._path + ".tmp", 'wb') as f:
                pickle.dump(kwargs, f, -1)
            self._replace_file(self._path + ".tmp", self._path)
        if self._requested:
            print "Checkpoint: terminating execution"
            sys.exit(9)

    def load(self):
        print "Checkpoint: %s" % self._path
        with open(self._path, 'rb') as f:
            return pickle.load(f)

    def remove(self):
        if os.path.exists(self._path):
            os.remove(self._path)

    @property
    def exists(self):
        return isfile(self._path)

