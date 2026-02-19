from __future__ import annotations
from typing import TYPE_CHECKING

from pygenn import SynapseMatrixType
from .sparse_base import SparseBase
from ..utils.snippet import ConnectivitySnippet
from ..utils.value import InitValue

from pygenn import (create_sparse_connect_init_snippet, init_sparse_connectivity)

if TYPE_CHECKING:
    from .. import Connection, Population
    from ..compilers.compiler import SupportedMatrixType


genn_snippet = create_sparse_connect_init_snippet(
    "toroidal_gaussian_2d",

    params=[("sigma", "scalar"), 
            ("p_max", "scalar"),
            ("src_h", "int"), ("src_w", "int"), ("src_c", "int"),
            ("tgt_h", "int"), ("tgt_w", "int"), ("tgt_c", "int"),
            ("self_connect", "int")],

    row_build_code=
        """
        const int inRow = (id_pre / src_c) / src_w;
        const int inCol = (id_pre / src_c) % src_w;
        const int inChan = id_pre % src_c;
        
        const float x1 = (float)inCol / (src_w - 1);
        const float y1 = (float)inRow / (src_h - 1);

        for(int outRow = 0; outRow < tgt_h; outRow++) {
            const float y2 = (float)outRow / (tgt_h - 1);
            
            for(int outCol = 0; outCol < tgt_w; outCol++) {
                const float x2 = (float)outCol / (tgt_w - 1);
                
                for(int outChan = 0; outChan < tgt_c; outChan++) {
                    const int idPost = ((outRow * tgt_w * tgt_c) +
                                        (outCol * tgt_c) +
                                        outChan);
                    
                    if(self_connect == 0 && id_pre == idPost) continue;
                    
                    float dx = fabs(x1 - x2);
                    dx = fmin(dx, 1.0f - dx);

                    float dy = fabs(y1 - y2);
                    dy = fmin(dy, 1.0f - dy);

                    const float d2 = dx*dx + dy*dy;

                    const float prob = p_max * exp(-d2 / (2.0f * sigma * sigma));

                    if (gennrand_uniform() < prob) {
                        addSynapse(idPost);
                    }
                }
            }
        }
        """)


class ToroidalGaussian2D(SparseBase):
    """
    Sparse Gaussian distance-dependent connectivity
    with periodic (toroidal) boundary conditions.
    """

    def __init__(self,
                 sigma: float,
                 p_max: float,
                 weight: InitValue,
                 allow_self_connections: bool = False,
                 delay: InitValue = 0):

        super(ToroidalGaussian2D, self).__init__(weight, delay)

        self.sigma = sigma
        self.p_max = p_max
        self.allow_self_connections = allow_self_connections

    def connect(self, source: Population, target: Population):
        # Store shapes with channel dimension
        if len(source.shape) == 3:
            self.src_h, self.src_w, self.src_c = source.shape
        else:
            self.src_h, self.src_w = source.shape
            self.src_c = 1
            
        if len(target.shape) == 3:
            self.tgt_h, self.tgt_w, self.tgt_c = target.shape
        else:
            self.tgt_h, self.tgt_w = target.shape
            self.tgt_c = 1

        # Check if recurrent
        self.is_recurrent = (source == target)

    def get_snippet(self,
                    connection: Connection,
                    supported_matrix_type: SupportedMatrixType):

        conn_init = init_sparse_connectivity(genn_snippet, {
            "sigma": self.sigma,
            "p_max": self.p_max,
            "src_h": self.src_h,
            "src_w": self.src_w,
            "src_c": self.src_c,
            "tgt_h": self.tgt_h,
            "tgt_w": self.tgt_w,
            "tgt_c": self.tgt_c,
            "self_connect": 1 if self.allow_self_connections or \
                connection.source() != connection.target() else 0
        })

        return super(ToroidalGaussian2D, self)._get_snippet(
            supported_matrix_type,
            conn_init
        )