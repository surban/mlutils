from os import makedirs
from socket import gethostname
import ctypes
from inspect import getargspec
import os
import sys
import glob
import time
import imp
import pickle
import signal
import itertools
import __main__ as main
from os.path import isfile, exists, split, abspath, isdir, join
import climin
from argparse import ArgumentParser


def multiglob(*patterns):
    return itertools.chain.from_iterable(
        glob.glob(pattern) for pattern in patterns)


def base_dir_of(p):
    while not os.path.isdir(os.path.join(p, "cfgs")):
        p = os.path.join(p, "..")
        if len(p) > 1000:
            raise RuntimeError("Cannot find base directory (contains a 'cfgs' subdirectory)")
    return p

def base_dir():
    """Finds the path to the base directory of the main module.
     The base directory is the first parent directory of the main module or,
     if no main module is loaded, the current directory, that contains a 'cfgs' subdirectory.
    :returns: path to the base directory"""
    try:
        p = os.path.dirname(main.__file__)
        if p.endswith("mlutils"):
            p = ""
    except AttributeError:
        p = ""
    return base_dir_of(p)

def cfg_base_dir():
    """
    Finds the path to the base directory of the currently parsed configuration file.
    :return:
    """
    return base_dir_of(cfg_dir())

def cfgs_dir():
    """Finds the path to the configuration directory.
    :returns: path to the configuration directory"""
    return os.path.join(base_dir(), "cfgs")

_cfg_dir = None
def cfg_dir():
    """Finds the directory of the currently parsed configuration file.
    :returns: path to directory of current cfg.py"""
    return _cfg_dir

def load_cfg(config_name=None, prepend="", clean_outputs=False, with_checkpoint=False, defaults={}, force_restart=False):
    """Reads the configuration file cfg.py from the configuration directory
    specified as the first parameter on the command line.
    Returns a tuple consisting of the configuration module and the plot directory.
    :param config_name: Configuration to load (can be specified relative to the cfgs directory).
                        If not specified, the command line is parsed to get the configuration file.
    :param prepend: Prepend subdirectory for config file. Use %SCRIPTNAME% to insert name of running script.
    :param clean_outputs: plot and data files are removed from the config directory
    :param with_checkpoint: enables checkpoint support
    :param defaults: default values for non-specified configuration variables
    :param forece_restart: if True, checkpoint loading is inhibited.
    :returns: if with_checkpoint == True:  (cfg module, cfg directory, checkpoint handler, checkpoint)
              if with_checkpoint == False: (cfg module, cfg directory)
    """
    global _cfg_dir
    outdir = None
    force_continue = False

    # gather information
    print "Host: %s" % gethostname()
    try:
        scriptname, scriptext = os.path.splitext(os.path.basename(main.__file__))
        print "Script: %s" % (scriptname + scriptext)
    except AttributeError:
        scriptname = "unknown"
    prepend = prepend.replace("%SCRIPTNAME%", scriptname)

    # parse command line arguments if cfg was not specified as function parameter
    if config_name is None:
        parser = ArgumentParser()
        parser.add_argument('cfg', help="configuration to load (can be specified relative to the cfgs directory)")
        parser.add_argument('--out-dir', help="output directory (by default config directory is used)")
        if with_checkpoint:
            parser.add_argument('--restart', action='store_true', help="inhibits loading of an available checkpoint")
            parser.add_argument('--continue', action='store_true', dest='cont', help="resets the termination criteria")

        args = parser.parse_args()
        config_name = args.cfg
        if args.out_dir is not None:
            outdir = args.out_dir
        if with_checkpoint:
            force_restart = args.restart
            force_continue = args.cont

    # determine path to configuration file
    if isfile(config_name):
        # path to config file was specified
        cfgname = abspath(config_name)
        cfgdir, _ = split(cfgname)
    elif isfile(join(config_name, 'cfg.py')):
        # path to config directory was specified
        cfgdir = abspath(config_name)
        cfgname = join(cfgdir, 'cfg.py')
    else:
        # treat specified configuration as relative to the configuration root directory
        cfgdir = abspath(join(cfgs_dir(), prepend, config_name))
        cfgname = join(cfgdir, 'cfg.py')

    # load configuration file
    if not os.path.exists(cfgname):
        raise RuntimeError("Config: %s not found" % cfgname)
    print "Config: %s" % cfgname
    sys.dont_write_bytecode = True
    _cfg_dir = cfgdir
    cfg = imp.load_source('cfg', cfgname)

    # prepare output directory
    if outdir is None:
        outdir = cfgdir
    print "Output directory: %s" % outdir
    if not exists(outdir):
        makedirs(outdir)

    # load checkpoint if requested
    checkpoint = None
    if with_checkpoint:
        cp_handler = CheckpointHandler(outdir)
        if (cp_handler.exists and not force_restart):
            checkpoint = cp_handler.load()
        else:
            print "Checkpoint: none"
            cp_handler.remove()

    # set defaults
    for k, v in defaults.iteritems():
        if k not in dir(cfg):
            setattr(cfg, k, v)
    def get_func(name):
        return getattr(cfg, name, default=None)
    setattr(cfg, 'get', get_func)

    # set additional information
    setattr(cfg, 'out_dir', outdir)
    setattr(cfg, 'continue_training', force_continue)

    # clean configuration directory
    if clean_outputs and checkpoint is None:
        curdir = os.path.abspath(os.curdir)
        os.chdir(outdir)
        for file in multiglob('*.png', '*.pdf', '*.npz'):
            os.remove(file)
        os.chdir(outdir)

    if with_checkpoint:
        return cfg, outdir, cp_handler, checkpoint
    else:
        return cfg, outdir


def optimizers_from_cfg(cfg, wrt_fprime_for_name, f, print_config=True):
    def optimizer_instance(name):
        # get specified optimizer and its constructor
        wrt, fprime = wrt_fprime_for_name(name)
        class_name = 'climin.' + cfg.optimizer[name]
        class_obj = eval(class_name)
        init_func = class_obj.__init__

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
                    kwargs[arg] = getattr(cfg, cfg_arg)[name]

        if print_config:
            argstr = ", ".join(["%s=%s" % (arg, str(value)) for arg, value in kwargs.iteritems()
                                if arg not in ['wrt', 'f', 'fprime']])
            print "optimizer for %s: %s (%s)" % (name, cfg.optimizer[name], argstr)

        return class_obj(**kwargs)

    return {name: optimizer_instance(name) for name in cfg.optimizer}


def optimizer_from_cfg(cfg, wrt, f, fprime, print_config=True):
    # get specified optimizer and its constructor
    class_name = 'climin.' + cfg.optimizer
    class_obj = eval(class_name)
    init_func = class_obj.__init__

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

    if print_config:
        argstr = ", ".join(["%s=%s" % (arg, str(value)) for arg, value in kwargs.iteritems()
                            if arg not in ['wrt', 'f', 'fprime']])
        print "optimizer: %s (%s)" % (cfg.optimizer, argstr)

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
            #ctypes.CDLL(os.path.join(basepath, 'core', 'libmmd.dll'))
            #ctypes.CDLL(os.path.join(basepath, 'core', 'libifcoremd.dll'))

            # install win32 console control event handler
            import win32api
            win32api.SetConsoleCtrlHandler(self._console_ctrl_handler, 1)
        else:
            # install unix signal handlers
            signal.signal(signal.SIGINT, self._signal_handler)
            signal.signal(signal.SIGBREAK, self._signal_handler)
        self._handler_installed = True

    def _signal_handler(self, signum, frame):
        print "Checkpoint: requested by termination signal"
        self._requested = True

    def release(self):
        if not self._handler_installed:
            return
        if sys.platform == 'win32':
            # WORKAROUND: Removing the handler causes and exception, thus we just leave it installed.
            #             But since self._handler_installed is set to False, the handler will chain to the OS handler.
            pass
        else:
            signal.signal(signal.SIGINT, None)
            signal.signal(signal.SIGBREAK, None)
        self._handler_installed = False

    def _console_ctrl_handler(self, ctrl_type):
        if self._handler_installed and ctrl_type in [0, 1]:  # Ctrl-C or Ctrl-Break
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
            # save time the checkpoint was created
            kwargs['save_time'] = time.time()

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

