import os
import sys
import glob
import imp
import pickle
import signal
import itertools
import __main__ as main


def multiglob(*patterns):
    return itertools.chain.from_iterable(
        glob.glob(pattern) for pattern in patterns)


def load_cfg(clean_plots=False, prepend_scriptname=True, with_checkpoint=False):
    """Reads the configuration file cfg.py from the configuration directory
    specified as the first parameter on the command line.
    Returns a tuple consisting of the configuration module and the plot
    directory."""
    if len(sys.argv) < 2:
        if with_checkpoint:
            print "Usage: %s <config> [continue]" % sys.argv[0]
        else:
            print "Usage: %s <config>" % sys.argv[0]
        sys.exit(1)

    scriptname, _ = os.path.splitext(os.path.basename(main.__file__))
    if prepend_scriptname:
        cfgdir = os.path.join(scriptname, sys.argv[1])
    else:
        cfgdir = sys.argv[1]
    cfgname = os.path.join(cfgdir, 'cfg.py')
    if not os.path.exists(cfgname):
        print "Configuration %s not found" % cfgname
        sys.exit(2)
    print "Using configuration %s" % cfgname
    sys.dont_write_bytecode = True
    cfg = imp.load_source('cfg', cfgname)

    # load checkpoint if requested
    checkpoint = None
    if with_checkpoint:
        cp_handler = CheckpointHandler(cfgdir)
        if (('JOB_REQUEUED' in os.environ and os.environ[
            'JOB_REQUEUED'] == 'yes') or
                (len(sys.argv) >= 3 and sys.argv[2].startswith("cont"))):
            checkpoint = cp_handler.load()
        else:
            print "Using no checkpoint"
            cp_handler.remove()

    # clean plot directory
    if clean_plots and checkpoint is None:
        curdir = os.path.abspath(os.curdir)
        os.chdir(cfgdir)
        for file in multiglob('*.png', '*.pdf'):
            os.remove(file)
        os.chdir(curdir)

    if with_checkpoint:
        return cfg, cfgdir, cp_handler, checkpoint
    else:
        return cfg, cfgdir


class CheckpointHandler(object):
    def __init__(self, directory, filename="checkpoint.dat"):
        self._path = os.path.join(directory, filename)
        self._directory = directory
        self._requested = False

        signal.signal(signal.SIGINT, self._signal_handler)

    def _signal_handler(self, signum, frame):
        self._requested = True

    @staticmethod
    def _replace_file(src, dest):
        if sys.platform == 'win32':
            if os.path.exists(dest):
                os.remove(dest)
            assert not os.path.exists(
                dest), "%s still exists after deleting it" % dest
        os.rename(src, dest)

    @property
    def requested(self):
        return self._requested

    def save(self, **kwargs):
        explicit = False
        if 'explicit' in kwargs:
            explicit = kwargs['explicit']

        if self._requested:
            print "Saving checkpoint %s" % self._path
        if self._requested or explicit:
            with open(self._path + ".tmp", 'wb') as f:
                pickle.dump(kwargs, f, -1)
            self._replace_file(self._path + ".tmp", self._path)
        if self._requested:
            print "Checkpoint saved. Exiting."
            sys.exit(9)

    def load(self):
        print "Loading checkpoint %s" % self._path
        with open(self._path, 'rb') as f:
            return pickle.load(f)

    def remove(self):
        if os.path.exists(self._path):
            os.remove(self._path)
