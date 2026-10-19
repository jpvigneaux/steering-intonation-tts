"""Probing neural audio representations for intonational tune structure (vendored subset).

Three modules of the code accompanying *Intonational tunes in neural audio representations*,
included here unchanged so that this repository installs on its own (MIT licence, see
``LICENSE`` in this directory):

``audio``       encoder wrappers -- audio file to frame sequence (the Mimi wrapper is used here)
``probing``     feature extractors, probe models, and the training loop
``config``      filesystem layout, read from a YAML file the user supplies
"""

__version__ = "1.0.0"

__all__ = ["audio", "config", "probing"]
