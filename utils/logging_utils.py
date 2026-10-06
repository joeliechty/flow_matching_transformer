import os
import subprocess
import sys


def git_commit():
    """Short hash of the checked-out commit ('-dirty' if tracked files have uncommitted
    changes), or 'unknown' outside a git checkout. Stored with results so every number
    traces back to the code that produced it."""
    repo = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    def git(*args):
        return subprocess.run(['git', *args], cwd=repo, capture_output=True, text=True,
                              check=True).stdout.strip()

    try:
        sha = git('rev-parse', '--short', 'HEAD')
        dirty = git('status', '--porcelain', '--untracked-files=no')
    except (OSError, subprocess.CalledProcessError):
        return 'unknown'
    return f"{sha}-dirty" if dirty else sha


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
