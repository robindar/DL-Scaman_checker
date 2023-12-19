import sys
import numpy as np

def pretty_wrapped(func):
    def wrapped(*args, **kwargs):
        try:
            res = func(*args, **kwargs)
            return f"[DONE] {res}"
        except AssertionError as e:
            return f"[FAIL] {e.__str__()}"
    return wrapped

def check_install():
    return "[DONE] Install ok"

@pretty_wrapped
def check_dimensions(x, y):
    assert isinstance(x, np.ndarray), f"Got X with type '{type(x)}', expected np.ndarray"
    assert isinstance(y, np.ndarray), f"Got y with type '{type(x)}', expected np.ndarray"
    assert x.ndim == 2, f"Got X with {x.ndim} dimensions, expected 2"
    assert y.ndim == 2, f"Got y with {y.ndim} dimensions, expected 2"
    assert x.shape == (100, 2), f"Got X with shape {x.shape}, expected (100,2)"
    assert y.shape == (100, 2), f"Got y with shape {y.shape}, expected (100,2)"
    return "Dimensions ok"
