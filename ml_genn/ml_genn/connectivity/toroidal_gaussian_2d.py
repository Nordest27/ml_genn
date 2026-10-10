from __future__ import annotations
from typing import TYPE_CHECKING, Optional

import numpy as np
import time

from pygenn import SynapseMatrixType
from .sparse_base import SparseBase
from ..utils.value import InitValue, ValueDescriptor

if TYPE_CHECKING:
    from .. import Connection, Population
    from ..compilers.compiler import SupportedMatrixType


class ToroidalGaussian2D(SparseBase):
    """
    Sparse Gaussian distance-dependent connectivity
    with periodic (toroidal) boundary conditions.

    Connectivity is generated entirely on the CPU and
    passed explicitly via (pre_ind, post_ind).

    exact=False (default) keeps the original sampler, so earlier results reproduce exactly. It draws
    ``fan_in`` Gaussian positions and drops duplicates, so the realised fan-in is far below ``fan_in``
    when sigma is about a grid cell (e.g. 45-147 instead of 300 in the Snake network); its coordinates
    are scaled by 1 / (width - 1), so the first and last rows/columns coincide on the torus and get
    fewer connections; and it removes the pre index equal to the post index even between different
    populations.
    exact=True samples exactly ``fan_in`` distinct sources (or all eligible ones if fewer), without
    replacement, with probabilities given by a Gaussian of the toroidal distance between cell centres
    ((i + 0.5) / n, wrapping at 1); self-connections are excluded only when source is target.

    p_max (distance-dependent connectivity): every source/target pair is connected independently with probability
    p_max * exp(-d^2 / (2 sigma^2)), d the toroidal distance between cell centres. The fan-in is a consequence of
    the neighbourhood (expected fan-in ~ p_max * 2 pi sigma^2 * channels, in cells) instead of a target that may
    force connections far into the Gaussian tails; ``fan_in`` is ignored. sigma_cells gives sigma in cells of the
    source grid (the same locality whatever the layer size); otherwise sigma is a fraction of the grid.
    sigma_ref="coarser" measures sigma_cells in cells of the coarser of the two grids (per axis): from a big to a small
    layer each target then pools its own share of the source (every source is used; the neighbourhood is sigma_cells
    of the target grid), from small to big it is unchanged. p_global adds a uniform probability for every pair
    (sparse random long-range connections on top of the local ones: a small-world graph). "source" (default) keeps sigma in source cells, so a big
    to small connection subsamples (at ratios above ~3 part of the source connects to nothing).
    """

    def __init__(self,
                 sigma: float,
                 fan_in: float,
                 weight: InitValue,
                 allow_self_connections: bool = False,
                 delay: InitValue = 0,
                 fan_in_scale: Optional[float] = None,
                 fan_in_scale_center: tuple[int, int] = (0.5, 0.5),
                 exact: bool = False,
                 p_max: Optional[float] = None,
                 sigma_cells: Optional[float] = None,
                 sigma_ref: str = "source",
                 p_global: float = 0.0):

        super(ToroidalGaussian2D, self).__init__(weight, delay)

        self.sigma = float(sigma)
        self.fan_in = fan_in
        self.fan_in_scale = fan_in_scale
        self.fan_in_scale_center = fan_in_scale_center
        self.allow_self_connections = allow_self_connections
        self.exact = exact
        self.p_max = p_max
        self.sigma_cells = sigma_cells
        if sigma_ref not in ("source", "coarser"):
            raise ValueError(f"sigma_ref must be 'source' or 'coarser', not {sigma_ref!r}")
        self.sigma_ref = sigma_ref
        self.p_global = p_global

    def _euclidean_distance(self, x1, y1, x2, y2):
        dx = x1 - x2
        dy = y1 - y2
        return np.sqrt(dx * dx + dy * dy)
    
    def _local_fan_in(self, post_row, post_col):
        if self.fan_in_scale is None:
            return int(self.fan_in)
        cx, cy = self.fan_in_scale_center
        d = np.sqrt((post_col - cx) ** 2 + (post_row - cy) ** 2)
        return int(np.round(self.fan_in / (1.0 + (d / self.fan_in_scale) ** 2.0)))

    def _connect_exact(self, source, target, src_shape, tgt_shape):
        src_h, src_w, src_c = src_shape
        tgt_h, tgt_w, tgt_c = tgt_shape
        n_src = src_h * src_w * src_c
        same = source is target
        src_rows = (np.arange(src_h) + 0.5) / src_h
        src_cols = (np.arange(src_w) + 0.5) / src_w
        two_s2 = 2.0 * self.sigma * self.sigma
        all_pre, all_post = [], []
        for r in range(tgt_h):
            y = (r + 0.5) / tgt_h
            dy = np.abs(src_rows - y)
            dy = np.minimum(dy, 1.0 - dy)
            for c in range(tgt_w):
                x = (c + 0.5) / tgt_w
                dx = np.abs(src_cols - x)
                dx = np.minimum(dx, 1.0 - dx)
                spatial = np.exp(-(dy[:, None] ** 2 + dx[None, :] ** 2) / two_s2).ravel()
                weights = np.repeat(spatial, src_c)          # index (row * w + col) * c + channel
                k = self._local_fan_in(y, x)
                for ch in range(tgt_c):
                    id_post = (r * tgt_w + c) * tgt_c + ch
                    w = weights
                    if same and not self.allow_self_connections:
                        w = weights.copy()
                        w[id_post] = 0.0
                    eligible = int(np.count_nonzero(w > 0))
                    n = min(k, eligible)
                    if n <= 0:
                        continue
                    pre_ids = np.random.choice(n_src, size=n, replace=False, p=w / w.sum())
                    all_pre.append(np.sort(pre_ids).astype(np.uint32))
                    all_post.append(np.full(n, id_post, dtype=np.uint32))
        return all_pre, all_post

    def _sigmas(self, src_hw, tgt_hw):
        """Gaussian width per axis (rows, cols) in source cells."""
        (src_h, src_w), (tgt_h, tgt_w) = src_hw, tgt_hw
        sig = self.sigma_cells if self.sigma_cells is not None else self.sigma * np.sqrt(src_h * src_w)
        if self.sigma_ref == "coarser":
            return sig * max(1.0, src_h / tgt_h), sig * max(1.0, src_w / tgt_w)
        return sig, sig

    @staticmethod
    def _toroidal_z2(src_hw, tgt_rc, sigmas):
        """Squared toroidal distance from a target cell centre to every source cell centre, in units of sigma."""
        (src_h, src_w), (y, x), (sig_y, sig_x) = src_hw, tgt_rc, sigmas    # y, x as fractions of the grid
        rows = np.abs((np.arange(src_h) + 0.5) - y * src_h)
        rows = np.minimum(rows, src_h - rows) / sig_y
        cols = np.abs((np.arange(src_w) + 0.5) - x * src_w)
        cols = np.minimum(cols, src_w - cols) / sig_x
        return rows[:, None] ** 2 + cols[None, :] ** 2

    def expected_fan_in(self, src_shape, tgt_shape):
        """Mean expected fan-in of the distance-dependent mode (for weight scaling)."""
        src_h, src_w, src_c = src_shape if len(src_shape) == 3 else (*src_shape, 1)
        tgt_h, tgt_w = tgt_shape[:2]
        sigmas = self._sigmas((src_h, src_w), (tgt_h, tgt_w))
        total = 0.0
        for r in range(tgt_h):
            for c in range(tgt_w):
                z2 = self._toroidal_z2((src_h, src_w), ((r + 0.5) / tgt_h, (c + 0.5) / tgt_w), sigmas)
                total += np.clip(self.p_max * np.exp(-z2 / 2) + self.p_global, 0, 1).sum()
        return src_c * total / (tgt_h * tgt_w)

    def _connect_bernoulli(self, source, target, src_shape, tgt_shape):
        src_h, src_w, src_c = src_shape
        tgt_h, tgt_w, tgt_c = tgt_shape
        sigmas = self._sigmas((src_h, src_w), (tgt_h, tgt_w))
        same = source is target
        all_pre, all_post = [], []
        for r in range(tgt_h):
            for c in range(tgt_w):
                z2 = self._toroidal_z2((src_h, src_w), ((r + 0.5) / tgt_h, (c + 0.5) / tgt_w), sigmas)
                p = np.repeat(np.clip(self.p_max * np.exp(-z2 / 2) + self.p_global, 0, 1).ravel(), src_c)
                for ch in range(tgt_c):
                    id_post = (r * tgt_w + c) * tgt_c + ch
                    keep = np.random.random(p.shape) < p
                    if same and not self.allow_self_connections:
                        keep[id_post] = False
                    pre_ids = np.flatnonzero(keep)
                    all_pre.append(pre_ids.astype(np.uint32))
                    all_post.append(np.full(len(pre_ids), id_post, dtype=np.uint32))
        return all_pre, all_post

    def connect(self, source, target):
        t = time.time()
        src_shape = source.shape
        tgt_shape = target.shape

        if len(src_shape) == 3:
            src_h, src_w, src_c = src_shape
        else:
            src_h, src_w = src_shape
            src_c = 1

        if len(tgt_shape) == 3:
            tgt_h, tgt_w, tgt_c = tgt_shape
        else:
            tgt_h, tgt_w = tgt_shape
            tgt_c = 1

        num_post = tgt_h * tgt_w * tgt_c

        if self.p_max is not None or self.exact:
            sampler = self._connect_bernoulli if self.p_max is not None else self._connect_exact
            all_pre, all_post = sampler(source, target, (src_h, src_w, src_c), (tgt_h, tgt_w, tgt_c))
            self.pre_ind = np.concatenate(all_pre).astype(np.int32)
            self.post_ind = np.concatenate(all_post).astype(np.int32)
            print("Init time", source, target, time.time() - t)
            return

        src_x_scale = 1.0 / max(src_w - 1, 1)
        src_y_scale = 1.0 / max(src_h - 1, 1)
        tgt_x_scale = 1.0 / max(tgt_w - 1, 1)
        tgt_y_scale = 1.0 / max(tgt_h - 1, 1)

        cx, cy = self.fan_in_scale_center
        d0 = self.fan_in_scale
        power = 2.0  # decay exponent (tune if needed)

        all_pre  = []
        all_post = []

        for id_post in range(num_post):
            post_spatial = id_post // tgt_c
            post_row = (post_spatial // tgt_w) * tgt_y_scale
            post_col = (post_spatial  % tgt_w) * tgt_x_scale

            # --------------------------------------------------
            # Deterministic spatial fan-in (power-law decay)
            # --------------------------------------------------
            if d0 is not None:
                d = np.sqrt(
                    (post_col - cx) ** 2 +
                    (post_row - cy) ** 2
                )

                local_fan_in = int(
                    np.round(
                        self.fan_in / (1.0 + (d / d0) ** power)
                    )
                )
            else:
                local_fan_in = int(self.fan_in)

            if local_fan_in <= 0:
                continue
            # --------------------------------------------------

            dy = np.random.normal(0, self.sigma, size=local_fan_in)
            dx = np.random.normal(0, self.sigma, size=local_fan_in)

            tx = (post_col + dx) % 1.0
            ty = (post_row + dy) % 1.0

            pre_col_idx = np.clip(
                np.round(tx / src_x_scale).astype(int), 0, src_w - 1
            )
            pre_row_idx = np.clip(
                np.round(ty / src_y_scale).astype(int), 0, src_h - 1
            )

            pre_chan = np.random.randint(0, src_c, size=local_fan_in)

            pre_ids = (pre_row_idx * src_w + pre_col_idx) * src_c + pre_chan

            if not self.allow_self_connections:
                pre_ids = pre_ids[pre_ids != id_post]

            pre_ids = np.unique(pre_ids)[:local_fan_in]

            all_pre.append(pre_ids.astype(np.uint32))
            all_post.append(
                np.full(len(pre_ids), id_post, dtype=np.uint32)
            )

        self.pre_ind  = np.concatenate(all_pre).astype(np.int32)
        self.post_ind = np.concatenate(all_post).astype(np.int32)

        print("Init time", source, target, time.time() - t)

    def get_snippet(self,
                    connection: Connection,
                    supported_matrix_type: SupportedMatrixType):
        # No snippet — indices are already provided
        return super(ToroidalGaussian2D, self)._get_snippet(
            supported_matrix_type,
            snippet=None
        )