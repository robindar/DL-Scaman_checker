import sys
import numpy as np

from dl_scaman_checker.common import __version__

def pretty_wrapped(func):
    def wrapped(*args, **kwargs):
        try:
            res = func(*args, **kwargs)
            done = "\033[92m{}\033[00m" .format("DONE")
            print(f"[{done}] {res}")
        except AssertionError as e:
            fail = "\033[91m{}\033[00m" .format("FAIL")
            print(f"[{fail}] {e.__str__()}")
    return wrapped

@pretty_wrapped
def check_install():
    return f"Install ok. Version is {__version__}"

@pretty_wrapped
def check_imports():
    assert "numpy" in sys.modules, "Module 'numpy' has not been imported."
    assert "matplotlib.pyplot" in sys.modules, "Module 'matplotlib.pyplot' has not been imported."
    return f"Imports ok"

@pretty_wrapped
def check_dimensions(x, y):
    n = 100
    assert isinstance(x, np.ndarray), f"Got X with type '{type(x)}', expected np.ndarray"
    assert isinstance(y, np.ndarray), f"Got y with type '{type(x)}', expected np.ndarray"
    assert x.ndim == 2, f"Got X with {x.ndim} dimensions, expected 2"
    assert y.ndim == 2, f"Got y with {y.ndim} dimensions, expected 2"
    assert x.shape == (n, 2), f"Got X with shape {x.shape}, expected ({n},2)"
    assert y.shape == (n, 2), f"Got y with shape {y.shape}, expected ({n},2)"
    return "Dimensions ok"
