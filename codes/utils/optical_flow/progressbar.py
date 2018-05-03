"""
Sebastian Raschka 2014-2016
Python Progress Indicator Utility
Author: Sebastian Raschka <sebastianraschka.com>
License: BSD 3 clause
Contributors: https://github.com/rasbt/pyprind/graphs/contributors
Code Repository: https://github.com/rasbt/pyprind
PyPI: https://pypi.python.org/pypi/PyPrind
"""

import time
import sys
import os
from io import UnsupportedOperation

try:
    import psutil
    psutil_import = True
except ImportError:
    psutil_import = False


class Prog():
    def __init__(self, iterations, track_time, stream, title,
                 monitor, update_interval=None):
        """ Initializes tracking object. """
        self.cnt = 0
        self.title = title
        self.max_iter = float(iterations)  # to support Python 2.x
        self.track = track_time
        self.start = time.time()
        self.end = None
        self.item_id = None
        self.eta = None
        self.total_time = 0.0
        self.last_time = self.start
        self.monitor = monitor
        self.stream = stream
        self.active = True
        self._stream_out = self._no_stream
        self._stream_flush = self._no_stream
        self._check_stream()
        self._print_title()
        self.update_interval = update_interval

        if monitor:
            if not psutil_import:
                raise ValueError('psutil package is required when using'
                                 ' the `monitor` option.')
            else:
                self.process = psutil.Process()
        if self.track:
            self.eta = 1

    def update(self, iterations=1, item_id=None, force_flush=False):
        """
        Updates the progress bar / percentage indicator.
        Parameters
        ----------
        iterations : int (default: 1)
            default argument can be changed to integer values
            >=1 in order to update the progress indicators more than once
            per iteration.
        item_id : str (default: None)
            Print an item_id sring behind the progress bar
        force_flush : bool (default: False)
            If True, flushes the progress indicator to the output screen
            in each iteration.
        """
        self.item_id = item_id
        self.cnt += iterations
        self._print(force_flush=force_flush)
        self._finish()

    def stop(self):
        """Stops the progress bar / percentage indicator if necessary."""
        self.cnt = self.max_iter
        self._finish()

    def _check_stream(self):
        """Determines which output stream (stdout, stderr, or custom) to use"""
        if self.stream:
            try:
                if self.stream == 1 and os.isatty(sys.stdout.fileno()):
                    self._stream_out = sys.stdout.write
                    self._stream_flush = sys.stdout.flush
                elif self.stream == 2 and os.isatty(sys.stderr.fileno()):
                    self._stream_out = sys.stderr.write
                    self._stream_flush = sys.stderr.flush

            # a fix for IPython notebook "IOStream has no fileno."
            except UnsupportedOperation:
                if self.stream == 1:
                    self._stream_out = sys.stdout.write
                    self._stream_flush = sys.stdout.flush
                elif self.stream == 2:
                    self._stream_out = sys.stderr.write
                    self._stream_flush = sys.stderr.flush
            else:
                if self.stream is not None and hasattr(self.stream, 'write'):
                    self._stream_out = self.stream.write
                    self._stream_flush = self.stream.flush
        else:
            print('Warning: No valid output stream.')

    def _elapsed(self):
        """ Returns elapsed time at update. """
        self.last_time = time.time()
        return self.last_time - self.start

    def _calc_eta(self):
        """ Calculates estimated time left until completion. """
        elapsed = self._elapsed()
        if self.cnt == 0 or elapsed < 0.001:
            return None
        rate = float(self.cnt) / elapsed
        self.eta = (float(self.max_iter) - float(self.cnt)) / rate

    def _calc_percent(self):
        """Calculates the rel. progress in percent with 2 decimal points."""
        return round(self.cnt / self.max_iter * 100, 2)

    def _no_stream(self, text=None):
        """ Called when no valid output stream is available. """
        pass

    def _get_time(self, _time):
        if (_time < 86400):
            return time.strftime("%H:%M:%S", time.gmtime(_time))
        else:
            s = (str(int(_time // 3600)) + ':' +
                 time.strftime("%M:%S", time.gmtime(_time)))
            return s

    def _finish(self):
        """ Determines if maximum number of iterations (seed) is reached. """
        if self.cnt >= self.max_iter:
            self.total_time = self._elapsed()
            self.end = time.time()
            self.last_progress -= 1  # to force a refreshed _print()
            self._print()
            if self.track:
                self._stream_out('\nTotal time elapsed: ' +
                                 self._get_time(self.total_time))
            self._stream_out('\n')
            self.active = False

    def _print_title(self):
        """ Prints tracking title at initialization. """
        if self.title:
            self._stream_out('{}\n'.format(self.title))
            self._stream_flush()

    def _print_eta(self):
        """ Prints the estimated time left."""
        self._calc_eta()
        self._stream_out(' | ETA: ' + self._get_time(self.eta))
        self._stream_flush()

    def _print_item_id(self):
        """ Prints an item id behind the tracking object."""
        self._stream_out(' | Item ID: %s' % self.item_id)
        self._stream_flush()

    def __repr__(self):
        str_start = time.strftime('%m/%d/%Y %H:%M:%S',
                                  time.localtime(self.start))
        str_end = time.strftime('%m/%d/%Y %H:%M:%S',
                                time.localtime(self.end))
        self._stream_flush()

        time_info = 'Title: {}\n'\
                    '  Started: {}\n'\
                    '  Finished: {}\n'\
                    '  Total time elapsed: '.format(self.title,
                                                    str_start,
                                                    str_end)\
                    + self._get_time(self.total_time)
        if self.monitor:
            try:
                cpu_total = self.process.cpu_percent()
                mem_total = self.process.memory_percent()
            except AttributeError:  # old version of psutil
                cpu_total = self.process.get_cpu_percent()
                mem_total = self.process.get_memory_percent()

            cpu_mem_info = '  CPU %: {:.2f}\n'\
                           '  Memory %: {:.2f}'.format(cpu_total, mem_total)

            return time_info + '\n' + cpu_mem_info
        else:
            return time_info

    def __str__(self):
        return self.__repr__()


class ProgressBar(Prog):
    """
    Initializes a progress bar object that allows visuzalization
    of an iterational computation in the standard output screen.
    Parameters
    ----------
    iterations : `int`
        Number of iterations for the iterative computation.
    track_time : `bool` (default: `True`)
        Prints elapsed time when loop has finished.
    stream : `int` (default: 2).
        Setting the output stream.
        Takes `1` for stdout, `2` for stderr, or a custom stream object
    title : `str` (default: `''`).
        Setting a title for the percentage indicator.
    monitor : `bool` (default: `False`)
        Monitors CPU and memory usage if `True` (requires `psutil` package).
    update_interval : float or int (default: `None`)
        The update_interval in seconds controls how often the progress
        is flushed to the screen.
        Automatic mode if `update_interval=None`.
    """
    def __init__(self, iterations, track_time=True,
                 stream=2, title='', monitor=False, update_interval=None):
        Prog.__init__(self, iterations, track_time, stream,
                      title, monitor, update_interval)
        self.last_progress = 0
        self._print()
        if monitor:
            try:
                self.process.cpu_percent()
                self.process.memory_percent()
            except AttributeError:  # old version of psutil
                self.process.get_cpu_percent()
                self.process.get_memory_percent()

    def _print(self, force_flush=False):
        """ Prints formatted percentage and tracked time to the screen."""
        next_perc = self._calc_percent()
        if self.update_interval:
            do_update = time.time() - self.last_time >= self.update_interval
        elif force_flush:
            do_update = True
        else:
            do_update = next_perc > self.last_progress

        if do_update and self.active:
            self.last_progress = next_perc
            self._stream_out('\r[%3d %%]' % (self.last_progress))
            if self.track:
                self._stream_out(' Time elapsed: ' +
                                 self._get_time(self._elapsed()))
                self._print_eta()
            if self.item_id:
                self._print_item_id()
#End of class ProgressBar
