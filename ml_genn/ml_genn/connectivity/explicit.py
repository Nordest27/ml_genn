from __future__ import annotations

import numpy as np

from typing import TYPE_CHECKING
from .sparse_base import SparseBase

if TYPE_CHECKING:
    from .. import Connection, Population
    from ..compilers.compiler import SupportedMatrixType


class Explicit(SparseBase):
    """Sparse connectivity given explicitly as index arrays: synapse i connects source neuron ``pre_ind[i]`` (flat
    index) to target neuron ``post_ind[i]`` with weight ``weight[i]`` (an array, or one value for all).

    Args:
        pre_ind:    Flat source-neuron index of every synapse
        post_ind:   Flat target-neuron index of every synapse
        weight:     Weight of every synapse (array of the same length) or a constant
        delay:      Connection delays
    """
    def __init__(self, pre_ind, post_ind, weight, delay=0):
        super(Explicit, self).__init__(weight, delay)
        self.pre_ind = np.asarray(pre_ind, dtype=np.int64)
        self.post_ind = np.asarray(post_ind, dtype=np.int64)
        if self.pre_ind.shape != self.post_ind.shape:
            raise ValueError("Explicit: pre_ind and post_ind must have the same length")

    def connect(self, source: Population, target: Population):
        if target.shape is None:
            raise RuntimeError("Explicit connectivity needs the target population's shape")
        if self.pre_ind.max(initial=0) >= np.prod(source.shape) or self.post_ind.max(initial=0) >= np.prod(target.shape):
            raise ValueError("Explicit: an index exceeds its population's size")

    def get_snippet(self, connection: Connection, supported_matrix_type: SupportedMatrixType):
        return super(Explicit, self)._get_snippet(supported_matrix_type, None)
