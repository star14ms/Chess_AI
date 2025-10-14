from rich import print as pprint


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
