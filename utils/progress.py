import re
from pathlib import Path

from rich import print as pprint

_ANSI_PATTERN = re.compile(r"\x1b\[[0-9;]*[a-zA-Z]?")


class LoggingProgressWrapper:
    """Wraps Progress/NullProgress so progress.print() also writes to train.log.
    Rich's Live display bypasses sys.stdout, so tee cannot capture progress.print()."""

    def __init__(self, progress, log_path):
        self._progress = progress
        self._log_path = Path(log_path) if log_path else None

    def __getattr__(self, name):
        return getattr(self._progress, name)

    def __setattr__(self, name, value):
        if name in ("_progress", "_log_path"):
            object.__setattr__(self, name, value)
        else:
            setattr(self._progress, name, value)

    def print(self, *args, **kwargs):
        self._progress.print(*args, **kwargs)
        if self._log_path:
            try:
                parts = []
                for a in args:
                    if hasattr(a, "plain"):
                        parts.append(str(a.plain))
                    else:
                        parts.append(str(a))
                text = _ANSI_PATTERN.sub("", " ".join(parts)).rstrip() + "\n"
                if text.strip():
                    with open(self._log_path, "a", encoding="utf-8") as f:
                        f.write(text)
                        f.flush()
            except Exception:
                pass


class NullProgress:
    def __init__(self, rich=True):
        self.rich = rich
    def start(self):
        pass
    def stop(self):
        pass
    def print(self, *args, **kwargs):
        if self.rich:
            pprint(*args, **kwargs)
        else:
            print(*args, **kwargs)
    def add_task(self, *args, **kwargs):
        return 0
    def update(self, *args, **kwargs):
        pass
    def start_task(self, *args, **kwargs):
        pass
    @property
    def columns(self):
        return []
    @columns.setter
    def columns(self, value):
        pass
