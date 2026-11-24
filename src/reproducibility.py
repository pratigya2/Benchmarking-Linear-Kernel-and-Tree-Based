"""Run bookkeeping: log teeing, run metadata, warning accounting."""

import sys


class _Tee:
    """File-like object that writes to a console stream and a log file."""

    def __init__(self, stream, log_file):
        self._stream = stream
        self._log = log_file

    def write(self, text):
        self._stream.write(text)
        self._log.write(text)
        return len(text)

    def flush(self):
        self._stream.flush()
        self._log.flush()

    def __getattr__(self, name):
        # isatty, fileno, encoding, ... come from the console stream
        return getattr(self._stream, name)


def start_log(log_path):
    """Duplicate everything printed to stdout and stderr into ``log_path``.

    The log is line-buffered so it can be followed live (``tail -f``).
    Output of joblib worker processes goes straight to the inherited file
    descriptors and is therefore not captured here.
    """
    log_file = open(log_path, "w", encoding="utf-8", buffering=1)
    sys.stdout = _Tee(sys.stdout, log_file)
    sys.stderr = _Tee(sys.stderr, log_file)
    return log_file
