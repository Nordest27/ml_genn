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
    """

    def __init__(self,
                 sigma: float,
                 fan_in: float,
                 weight: InitValue,
                 allow_self_connections: bool = False,
                 delay: InitValue = 0,
                 fan_in_scale: Optional[float] = None,
                 fan_in_scale_center: tuple[int, int] = (0.5, 0.5)):

        super(ToroidalGaussian2D, self).__init__(weight, delay)

        self.sigma = float(sigma)
        self.fan_in = fan_in
        self.fan_in_scale = fan_in_scale
        self.fan_in_scale_center = fan_in_scale_center
        self.allow_self_connections = allow_self_connections

    def _euclidean_distance(self, x1, y1, x2, y2):
        dx = x1 - x2
        dy = y1 - y2
        return np.sqrt(dx * dx + dy * dy)
    
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