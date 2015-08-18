import time
import datetime
import sys

try:
    import IPython.core.display
    __IPYTHON__
    have_notebook = True
except (ImportError, NameError):
    have_notebook = False

_start_time = 0
_progress_interval = 0


def _clear_output():
    try:
        IPython.core.display.clear_output(stdout=False, stderr=False,
                                          other=True)
    except TypeError:
        IPython.core.display.clear_output(wait=True)


def status(current_iter, max_iter=0, caption=""):
    """Displays the progress of a loop as a progressbar.
    current_iter is the number of the current iteration and max_iter is the
    total number of iterations. caption is an optional caption to display."""
    global _start_time, _progress_interval

    if current_iter == 0:
        _start_time = time.time()
        time_left = ""

        _progress_interval = 0
    else:
        try:
            cur_time = time.time()
            iters_per_sec = (cur_time - _start_time) / current_iter
            secs_left = (max_iter - current_iter) * iters_per_sec
            d = datetime.datetime(1, 1, 1) + datetime.timedelta(
                seconds=secs_left)
            time_left = "%d:%02d:%02d left" % (d.hour, d.minute, d.second)
        except OverflowError:
            time_left = ""

        if _progress_interval == 0:
            _progress_interval = current_iter

    if current_iter > max_iter:
        time_left = "%d" % current_iter
    elif _progress_interval != 0 and current_iter + \
            _progress_interval >= max_iter:
        done()
        return

    if caption is None:
        return
    if len(caption) > 0:
        desc = "%s: " % caption
    else:
        desc = ""
    if have_notebook:
        _clear_output()
        if max_iter > 0:
            IPython.core.display.display_html(
                '<i>%s</i><meter value="%d" min="0" max="%d">%d / %d</meter> %s'
                % (desc, current_iter, max_iter, current_iter, max_iter,
                   time_left), raw=True)
        else:
            IPython.core.display.display_html(
                '<pre>%d: %s' % (current_iter, caption), raw=True)
    else:
        if max_iter > 0:
            print "%s%d / %d (%s)                                 \r" \
                  % (desc, current_iter, max_iter, time_left),
        else:
            print "%d: %s                                         \r" \
                  % (current_iter, caption),


def done():
    """Removes the progressbar when the loop is done. Calling is optional.
    In most cases the progressbar is automatically removed."""
    if have_notebook:
        _clear_output()
    else:
        print "                                                            \r",
