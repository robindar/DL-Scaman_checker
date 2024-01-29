import sys

from dl_scaman_checker.common import __version__

def pretty_warn(msg):
    warn = "\033[93m{}\033[00m" .format("WARN")
    print(f"[{warn}] {msg}")

def pretty_wrapped(func):
    def wrapped(*args, **kwargs):
        try:
            res = func(*args, **kwargs)
            done = "\033[92m{}\033[00m" .format("DONE")
            if res is not None:
                print(f"[{done}] {res}")
        except AssertionError as e:
            fail = "\033[91m{}\033[00m" .format("FAIL")
            print(f"[{fail}] {e.__str__()}")
    return wrapped

@pretty_wrapped
def check_install():
    return f"Install ok. Version is v{__version__}"

@pretty_wrapped
def check_imports():
    for m in [ "numpy", "matplotlib.pyplot", "torch" ]:
        assert m in sys.modules, f"Module '{m}' has not been imported."
    return f"Imports ok"

@pretty_wrapped
def check_dimensions(x, y):
    import torch
    n = 100
    assert isinstance(x, torch.Tensor), f"Got X with type '{type(x)}', expected torch.Tensor"
    assert isinstance(y, torch.Tensor), f"Got y with type '{type(x)}', expected torch.Tensor"
    assert x.ndim == 2, f"Got X with {x.ndim} dimensions, expected 2"
    assert y.ndim == 2, f"Got y with {y.ndim} dimensions, expected 2"
    assert x.shape == (n, 2), f"Got X with shape {x.shape}, expected ({n},2)"
    assert y.shape == (n, 2), f"Got y with shape {y.shape}, expected ({n},2)"
    return "Dimensions ok"

@pretty_wrapped
def check_distributions(x, y):
    import torch
    assert torch.max(torch.abs(x)) <= 2, f"Got X outside the [-2, +2] square"
    assert torch.max(torch.abs(y * (1-y))) < 1e-6, "Got y with values other than {0,1}, check your entries"
    assert torch.max(torch.abs(torch.sum(y,axis=1) - 1)) < 1e-6, "Each one-hot encoded vector should have values that sum to 1"
    assert ((y[:,0] - y[:,1]) * (x[:,0] * x[:,1] > 0) >= 0).all(), f"Found incorrect y coordinates, check the formula"
    return "Distributions ok"

@pretty_wrapped
def check_model_signature(data, model):
    import torch.nn as nn
    assert isinstance(model, nn.Sequential), "Model is not a torch.nn.Sequential model"
    x, y = data
    assert (x.ndim == 2) and (x.shape[1] == 2), "Got X with incorrect shape"
    z = model(x)
    assert z.ndim == 2, f"Got model output with {z.ndim} dimensions, but expected 2"
    assert z.shape == y.shape, f"Got model output with shape {z.shape}, but expected {y.shape}"
    return "Signature ok"

@pretty_wrapped
def check_accuracy(data, model, acc):
    import torch
    x, y = data
    z = model(x)
    a = (z[:,1] - z[:,0]) * (y[:,1] - y[:,0]) > 0
    real_acc = torch.mean(a.float())
    assert torch.abs(acc - real_acc) < 1e-3, f"Got a measured accuracy of {acc:.2f} but expected {real_acc:.2f}"
    return "Accuracy measure ok"


def visualize(data, model, epoch, losses):
    import torch
    import numpy as np
    import matplotlib.pyplot as plt
    from IPython.display import clear_output

    losses = list(map(lambda l: l.item() if isinstance(l, torch.Tensor) else l, losses))

    X, y = data
    xx, yy = np.meshgrid(np.linspace(-2.5, 2.5, 100), np.linspace(-2.5, 2.5, 100))

    grid_tensor = torch.FloatTensor(np.stack([xx.ravel(), yy.ravel()], axis=1))
    with torch.no_grad():
        Z = model(grid_tensor)
        Z = Z[:,0] - Z[:,1]

    Z = Z.reshape(xx.shape)

    fig = plt.figure(1, figsize=(10, 4))
    plt.subplot(1,2,1)
    CS = plt.contourf(xx, yy, Z, alpha=0.8)
    fig.colorbar(CS)
    plt.scatter(X[:, 0], X[:, 1], c=y[:,0])
    plt.xlim([-2,2])
    plt.ylim([-2,2])
    plt.xlabel("$X_1$")
    plt.ylabel("$X_2$")
    plt.title(f"Classification with an MLP (epoch = {epoch})")

    fmt = '-' if len(losses) > 100 else '-o'
    plt.subplot(1,2,2)
    plt.plot(losses, fmt)
    plt.yscale('log')
    plt.grid(alpha=0.5)
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Loss vs. number of epochs")
    plt.tight_layout()
    clear_output(wait=True)
    plt.show()
