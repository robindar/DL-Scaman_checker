import sys

from dl_scaman_checker.common import __version__
from .TP01 import pretty_wrapped, pretty_warn

import torch
import numpy as np
import matplotlib.pyplot as plt

@pretty_wrapped
def check_install():
    return f"Install ok. Version is v{__version__}"

@pretty_wrapped
def check_epochs(epochs):
    assert epochs is not None, f"You did not set the `NUM_EPOCH` constant."
    if epochs < 2:
        pretty_warn("You will not reach reasonable accuracies with less that 2 epochs. Increase the constant if possible")
    if epochs > 12:
        pretty_warn("You have set a very high number of epochs. Lower the constant unless you have confirmed acceptable running times.")


def downsample(inputs):
    return torch.nn.functional.avg_pool2d(inputs, kernel_size=4)

def plot_digits(ims, interp=False, titles=None):
    ims= [ np.array(im) for im in ims ]
    if np.all([im.ndim == 1 for im in ims]):
        return plot(np.array(ims), interp)
    mn,mx=min([im.min() for im in ims]),max([im.max() for im in ims])
    f = plt.figure(figsize=(12,24))
    plt.gray()
    for i, im in enumerate(ims):
        sp=f.add_subplot(1, len(ims), i+1)
        get = lambda t: t.item() if isinstance(t, torch.Tensor) else t
        if not titles is None: sp.set_title(get(titles[i]), fontsize=18)
        plt.imshow(im, interpolation=None if interp else 'none', vmin=mn,vmax=mx)

def plot(im, interp=False):
    f = plt.figure(figsize=(3,6), frameon=True)
    plt.gray()
    plt.imshow(im, interpolation=None if interp else 'none')


class ScoreKeeper():
    def __init__(self):
        self.scores = {}

    def clear(self, key):
        self.scores[key] = []

    def register(self, key, train, test):
        if not key in self.scores:
            self.scores[key] = []
        self.scores[key].append((train, test))

    def plot(self, zoom_in=False):
        registered_colors = {
            "Random features": "tab:red",
            "Random conv features": "tab:purple",
            "Manual + Random conv features": "tab:olive",
            "Linear": "tab:green",
            "Linear - downsampled": "tab:blue",
            "Manual convolutional": "tab:orange",
            "Conv 2-layer": "black",
            "Conv 4-layer": "grey",
            "Conv residual": "turquoise",
        }
        min_shown = 0.9 if zoom_in else 0.8
        transform = lambda u: np.maximum( np.array(u), min_shown )
        for key in self.scores:
            scores = np.array(self.scores[key])
            train, test = scores[:,0], scores[:,1]
            kwargs = { 'marker': 'o' }
            if key in registered_colors:
                kwargs['color'] = registered_colors[key]
            plt.plot(transform(train), transform(test), '-', label=key, **kwargs, zorder=12, clip_on=False)
        plt.grid(alpha=0.5, which='major')
        plt.grid(alpha=0.1, which='minor')
        plt.xlabel("Train accuracy")
        plt.ylabel("Test accuracy")
        ticks = [ i / 20 for i in range(1, 21) ]
        plt.xticks(transform(ticks), labels=[f"{int(100*t):02d}%" for t in ticks])
        plt.yticks(transform(ticks), labels=[f"{int(100*t):02d}%" for t in ticks])

        ticks = [ i / 100 for i in range(1, 101) ]
        plt.xticks(transform(ticks), labels=['']*len(ticks), minor=True)
        plt.yticks(transform(ticks), labels=['']*len(ticks), minor=True)

        plt.plot([0,1], [0,1], ':', color='#666666', linewidth=1)
        plt.tick_params(labeltop=True, top=True, labelright=True, right=True, which="both")

        plt.ylim(transform(min_shown), transform(1))
        plt.xlim(transform(min_shown), transform(1))
        plt.legend(loc='lower right')
        plt.show()
