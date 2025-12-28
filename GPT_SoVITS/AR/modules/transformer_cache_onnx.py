from typing import Optional
import torch
from torch import Tensor
import torch.nn as nn

@torch.jit.script
class TransformerCache:
    def __init__(
        self,
        all_stage: int,
        k: Tensor ,
        v: Tensor ,
        y_emb: Tensor ,
        first_infer: int = 1,
        stage: int = 0
    ):
        self.all_stage = all_stage
        self.k = k
        self.v = v
        self.y_emb = y_emb
        self.first_infer = first_infer
        self.stage = stage
