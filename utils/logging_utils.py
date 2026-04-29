import os
import sys

class _Tee:
    def __init__(self, log_path):
        os.makedirs(os.path.dirname(log_path) or '.', exist_ok=True)
        self._file = open(log_path, 'w')
        self._stdout = sys.stdout
        sys.stdout = self

    def write(self, data):
        self._stdout.write(data)
        self._file.write(data)

    def flush(self):
        self._stdout.flush()
        self._file.flush()

    def close(self):
        sys.stdout = self._stdout
        self._file.close()
