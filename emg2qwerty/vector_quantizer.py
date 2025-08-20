import abc

import math

import torch
from torch import nn


class BaseVectorQuantizer(nn.Module):
    def __init__(
        self,
        parent_block_id: int,
        embedding_dim: int,
    ) -> None:
        super().__init__()
        assert parent_block_id >= 0
        self.parent_block_id = parent_block_id
        self.embedding_dim = embedding_dim

    @abc.abstractmethod
    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    @torch.jit.export
    @abc.abstractmethod
    def compressed_bits(self) -> int:
        """Number of bits per timestep in the compressed stream."""
        raise NotImplementedError


class VectorQuantizer(BaseVectorQuantizer):
    def __init__(
        self,
        parent_block_id: int,
        embedding_dim: int,
        codebook_size: int,
    ) -> None:
        super().__init__(parent_block_id=parent_block_id, embedding_dim=embedding_dim)
        self.codebook_size = codebook_size
        self.codebook = nn.Embedding(codebook_size, embedding_dim)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        # inputs: TNC
        assert inputs.ndim == 3
        assert inputs.shape[2] == self.embedding_dim

        l2_dist = torch.cdist(inputs, self.codebook.weight, p=2.0)
        nearest_idx = l2_dist.argmin(dim=-1)
        quantized = self.codebook(nearest_idx)

        # Straight-through estimator
        quantized = (quantized - inputs).detach() + inputs

        return quantized

    @torch.jit.export
    def compressed_bits(self) -> int:
        return math.ceil(math.log2(self.codebook_size))
