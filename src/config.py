"""Configuration object.."""

import os

_dir = os.getcwd()

__config = {}
# Config = None


class _Config(dict):
    def __init__(self, *args, **kwargs):
        super(_Config, self).__init__(*args, **kwargs)
        self.__dict__ = self


def init_config(parser):
    """Initialise the global config variable.

    It holds all parameters for the configuration of the prototype.
    """
    global Config
    Config = _Config(parser.__dict__)
