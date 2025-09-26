

class NullProgress:
    def start(self):
        pass
    def stop(self):
        pass
    def print(self, *args, **kwargs):
        print(*args)
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
