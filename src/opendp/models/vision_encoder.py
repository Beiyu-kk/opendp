from __future__ import annotations

from torch import nn


class VisionEncoder(nn.Module):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__()
        raise NotImplementedError("Image observations are not implemented in this minimal build.")
