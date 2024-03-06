import sys

from dl_scaman_checker.common import __version__
from .TP01 import pretty_wrapped, pretty_warn

import torch
import torch.nn.functional as F

@pretty_wrapped
def check_install():
    return f"Install ok. Version is v{__version__}"


def gumbel_softmax(logits, tau=1, hard=False, eps=1e-10, dim=-1):
    # type: (Tensor, float, bool, float, int) -> Tensor
    r"""
    This is a stabilized version of PyTorch's `F.gumbel_softmax`
    function, which samples zeros from the exponential and the
    nans when taking the logarithm.
    """

    def stabilized_gumbel():
        gumbels = - torch.empty_like(logits).exponential_().log()
        reject = torch.isnan(gumbels).sum() or torch.isinf(gumbels).sum()
        return stabilized_gumbel() if reject else gumbels

    gumbels = stabilized_gumbel()  # ~Gumbel(0,1)
    gumbels = (logits + gumbels) / tau  # ~Gumbel(logits,tau)
    y_soft = gumbels.softmax(dim)

    if hard:
        # Straight through.
        index = y_soft.max(dim, keepdim=True)[1]
        y_hard = torch.zeros_like(logits).scatter_(dim, index, 1.0)
        ret = y_hard - y_soft.detach() + y_soft
    else:
        # Reparametrization trick.
        ret = y_soft
    return ret
