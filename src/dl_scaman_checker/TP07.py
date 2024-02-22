import sys

from dl_scaman_checker.common import __version__
from .TP01 import pretty_wrapped, pretty_warn

import torch
import numpy as np
from urllib.request import urlretrieve

@pretty_wrapped
def check_install():
    return f"Install ok. Version is v{__version__}"

def printable_parameter_count(model):
    count = sum(p.numel() for p in model.parameters())
    return human_readable_unit(count)

def human_readable_unit(integer):
    assert integer >= 0, f"Cannot compute human-readable form of negative integer '{integer}'"
    category = int(np.floor(np.log10(integer) / 3))
    cfg = [ ("", 0), ("K", 0), ("M", 1), ("G", 1), ("T", 2), ("P", 2) ]
    trail, rnd = cfg[category]
    return f"{np.round(integer / 10 ** (3*category), rnd):.{rnd}f}{trail}"

@pretty_wrapped
def download_tinyshakespeare(filename):
    url = (
        "https://raw.githubusercontent.com/"
        "karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
    )
    path, headers = urlretrieve(url, filename)
    ETag = '"a82bf4e8979c373a24f616ef6c044f821e18ce64322e5e1280f069f2910b3653"'
    sig = ""
    if ("ETag" in headers) and (headers["ETag"] == ETag):
        sig = ", ETag matches expectation"
    return f"Download ok{sig}. File available at '{filename}'"

class Alphabet:
    def __init__(self, alphabet):
        self.characters = alphabet

        self.stoi = { ch: i for i,ch in enumerate(alphabet) }
        self.itos = { i: ch for i,ch in enumerate(alphabet) }

        # for ch in alphabet:
        #     if ch not in allowed_chars:
        #         self.stoi[ch] = stoi[' ']

    def encode(self, s):
        # return torch.tensor([ self.stoi[c] for c in s ], dtype=torch.long)
        return [ self.stoi[c] for c in s ]

    def decode(self, l):
        if isinstance(l, torch.Tensor):
            l = l.detach().numpy()
        return "".join([ self.itos[i] for i in list(l) ])

def read_from_file(filename):
    with open(filename, 'r', encoding='utf-8') as f:
        text = f.read()

    chars = "".join(sorted(list(set(text))))
    alphabet = Alphabet(chars)

    data = torch.tensor(alphabet.encode(text), dtype=torch.long)
    n = int(0.9 * len(data))
    train_data = data[:n]
    val_data = data[n:]

    return alphabet, train_data, val_data


class SampleLogitModel():
    def __init__(self, alphabet, histsize=4):
        vocab_size = len(alphabet.characters)
        logits = np.ones(tuple([vocab_size] * (histsize+1)), dtype=np.float32) * 1e-6
        fakedata = alphabet.encode("The quick brown fox jumps over the lazy dog. " * 20)
        for i in range(histsize):
            tail = alphabet.encode("The qui")[:i+1]
            logits[tuple([Ellipsis] + list(tail))] += 10 ** i
        for i in range(len(fakedata) - histsize):
            logits[tuple(fakedata[i:i+histsize+1])] += 1e4
        self.logits, self.histsize = np.log(logits), histsize

    def __call__(self, seq):
        assert len(seq) >= self.histsize
        return self.logits[tuple(seq[-self.histsize:])]
