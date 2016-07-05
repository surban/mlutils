from subprocess import Popen, PIPE
import importlib
import os
from os.path import join
import json


def git_log(modules=None, log_dir=None, check=False):
    """
    Logs the current state (version of latest git commit) of the specified modules
    :param log_dir: directory where gitlog is saved
    :param modules: list of python modules for which version or latest git commit should be logged, defaults to theano,
        climin and mlutils if no modules are specified
    :param check: if True load the gitlog.json from the current directory and warn if there have been changes since that
        gitlog was created
    :return: the created log as a dictionary
    """
    if modules is None:
        modules = ['theano', 'climin', 'mlutils']
    prev_path = os.getcwd()
    commits = {}
    warnings = {}
    for m in modules:
        mod = importlib.import_module(m)
        path = os.path.dirname(mod.__file__)
        os.chdir(path)
        gitproc = Popen(['git', 'diff-index', 'HEAD', '--numstat'], stdout=PIPE, stderr=PIPE)
        (stdout, stderr) = gitproc.communicate()
        # if module is not a repo, check for version
        if stderr.strip().startswith('fatal'):
            if hasattr(mod, '__version__'):
                commits[m] = mod.__version__
            else:
                commits[m] = "no info found"
        else:
            info = stdout.strip()
            if not info == '':
                print 'WARNING: unsynced changes in module %s' % m
                warnings[m] = info
            gitproc = Popen(['git', 'log','-1', '--pretty=oneline'], stdout=PIPE)
            (stdout, _) = gitproc.communicate()
            commits[m] = stdout.strip()

    log = {"commits": commits, "warnings": warnings}
    os.chdir(prev_path)
    if log_dir is not None:
        log_filename = join(log_dir, 'gitlog.json')
        with open(log_filename, 'wb') as log_file:
            json.dump(log, log_file, indent=4)
    return log
