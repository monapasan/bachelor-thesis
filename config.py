import os

_dir = os.getcwd()

__config = {}
# Config = None


class _Config(dict):
    def __init__(self, *args, **kwargs):
        super(_Config, self).__init__(*args, **kwargs)
        self.__dict__ = self


def init_config(parser):
    global Config
    Config = _Config(parser.__dict__)
    # __config.update(parser.__dict__)

# bandwidth = win_size**2
# minRadius = 8 ?
