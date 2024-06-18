# =================================
# File : spd
# Author : LGX
# Description : 
# CREATE TIME : 2023/11/23 11:35
# =================================
# -*- coding: utf-8 -*-

import torch


class SPD(torch.nn.Module):
    # Changing the dimension of the Tensor
    def __init__(self, dimension=1):
        super().__init__()
        self.d = dimension

    def forward(self, x):
        return torch.cat([x[..., ::2, ::2], x[..., 1::2, ::2], x[..., ::2, 1::2], x[..., 1::2, 1::2]], 1)
