from __future__ import annotations
from typing import TYPE_CHECKING, Optional

import numpy as np

from .sparse_base import SparseBase
from ..utils.value import InitValue

if TYPE_CHECKING:
    from .. import Connection, Population
    from ..compilers.compiler import SupportedMatrixType


def local_block_permutation(h, w, radius, rng):
    """A random permutation of the cells of an h x w torus that moves every cell at most `radius` cells per axis:
    blocks of (radius + 1) x (radius + 1) cells, at a random offset, are shuffled. Returns perm (h, w): cell (y, x)
    is mapped to the flat cell index perm[y, x]."""
    b = int(radius) + 1
    oy, ox = rng.randint(0, b), rng.randint(0, b)
    cell = np.arange(h * w).reshape(h, w)
    perm = cell.copy()
    for y0 in range(0, h, b):
        for x0 in range(0, w, b):
            block = [((oy + y0 + dy) % h, (ox + x0 + dx) % w)
                     for dy in range(min(b, h - y0)) for dx in range(min(b, w - x0))]
            shuffled = rng.permutation([cell[y, x] for y, x in block])
            for (y, x), src in zip(block, shuffled):
                perm[y, x] = src
    return perm


def local_shift_field(h, w, radius, rng):
    """Displacement field of a local permutation of an h x w sheet: offsets[y, x] = (dy, dx), as fractions of the
    sheet, from cell (y, x) to the cell it is mapped to (toroidal, |d| <= radius cells per axis). Populations of other
    sizes on the same sheet use the offset of the sheet cell they lie on, so they are displaced together."""
    perm = local_block_permutation(h, w, radius, rng)
    ty, tx = np.divmod(perm, w)
    y, x = np.mgrid[0:h, 0:w]
    dy = (ty - y + h / 2) % h - h / 2
    dx = (tx - x + w / 2) % w - w / 2
    return np.stack([dy / h, dx / w], axis=-1)


class LocalPermutation(SparseBase):
    """One-to-one connectivity between two populations on the same (height, width, channels) grid that moves every
    cell at most ``radius`` cells per axis: a random local permutation (no mixing, nothing lost, the topography is
    reorganised). The grid is cut into blocks of (radius + 1) x (radius + 1) cells, with a random offset on the torus,
    and the cells of every block are shuffled; a cell's channels move together (channel c to channel c).

    Args:
        radius: largest displacement per axis, in cells (0: the identity)
        weight: connection weights (a constant or an initializer)
        seed:   random seed of the offset and the shuffles (None: numpy's global random state)
    """
    def __init__(self, radius: int, weight: InitValue, delay: InitValue = 0, seed: Optional[int] = None):
        super(LocalPermutation, self).__init__(weight, delay)
        self.radius = int(radius)
        self.seed = seed

    def connect(self, source: Population, target: Population):
        s, t = tuple(source.shape), tuple(target.shape)
        if len(s) != 3 or s != t:
            raise ValueError(f"LocalPermutation needs equal (height, width, channels) grids, not {s} and {t}")
        h, w, c = s
        rng = np.random if self.seed is None else np.random.RandomState(self.seed)
        perm = local_block_permutation(h, w, self.radius, rng)    # target cell (y, x) receives from cell perm[y, x]
        post_cells = np.repeat(np.arange(h * w), c)
        pre_cells = np.repeat(perm.ravel(), c)
        ch = np.tile(np.arange(c), h * w)
        self.pre_ind = (pre_cells * c + ch).astype(np.int32)
        self.post_ind = (post_cells * c + ch).astype(np.int32)

    def get_snippet(self, connection: Connection,
                    supported_matrix_type: SupportedMatrixType):
        return super(LocalPermutation, self)._get_snippet(supported_matrix_type, None)
