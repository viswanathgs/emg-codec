import abc

import logging
import math

import torch
from torch import nn

from emg2qwerty import utils

log = logging.getLogger(__name__)


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
    def nbits_compressed(self) -> int:
        """Number of bits per timestep in the compressed stream."""
        raise NotImplementedError


class VectorQuantizer(BaseVectorQuantizer):
    def __init__(
        self,
        parent_block_id: int,
        embedding_dim: int,
        codebook_size: int,
        kmeans_init: bool = True,
        kmeans_max_iters: int = 300,
        kmeans_rtol: float = 1e-6,
    ) -> None:
        super().__init__(parent_block_id=parent_block_id, embedding_dim=embedding_dim)

        self.codebook_size = codebook_size
        self.kmeans_init = kmeans_init
        self.kmeans_max_iters = kmeans_max_iters
        self.kmeans_rtol = kmeans_rtol

        self.codebook = nn.Embedding(codebook_size, embedding_dim)

        # Whether the codebook has been initialized
        self.register_buffer("_initialized", torch.tensor(False))

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        # inputs: TNC
        assert inputs.ndim == 3
        assert inputs.shape[2] == self.embedding_dim

        # Lazy-init codebook entries based on the first training batch.
        # No-op if kmeans_init is False.
        if self.training and not self._initialized:
            self._init_codebook(inputs)
            self._initialized.fill_(True)

        l2_dist = torch.cdist(inputs, self.codebook.weight, p=2.0)
        nearest_idx = l2_dist.argmin(dim=-1)
        quantized = self.codebook(nearest_idx)

        # Straight-through estimator
        quantized = (quantized - inputs).detach() + inputs

        return quantized

    @torch.jit.export
    def nbits_compressed(self) -> int:
        return math.ceil(math.log2(self.codebook_size))

    @torch.jit.ignore
    def _init_codebook(self, inputs: torch.Tensor) -> None:
        if not self.kmeans_init:
            # Fallback to default random init for codebook entries
            return

        # Run kmeans only on rank 0
        if utils.get_rank() == 0:
            log.info("Initializing codebook using k-means clustering on rank 0 batch")
            centroids = self._kmeans(
                samples=inputs.view(-1, self.embedding_dim),
                n_clusters=self.codebook_size,
                max_iters=self.kmeans_max_iters,
                rtol=self.kmeans_rtol,
            )
        else:
            centroids = torch.zeros(
                self.codebook_size,
                self.embedding_dim,
                dtype=inputs.dtype,
                device=inputs.device,
            )

        # Broadcast cluster centroids from rank 0 to the rest
        utils.broadcast_tensor(centroids, src_rank=0)

        # Init codebook entries with cluster centroids
        with torch.no_grad():
            self.codebook.weight.copy_(centroids)

    @torch.jit.ignore
    def _kmeans(
        self,
        samples: torch.Tensor,
        n_clusters: int,
        max_iters: int = 300,
        rtol: float = 1e-6,
    ) -> torch.Tensor:
        assert samples.ndim == 2

        n_samples = len(samples)
        assert n_samples >= n_clusters

        # Random centroid init.
        # TODO: switch to k-means++ init.
        centroids_idx = torch.randperm(n_samples)[:n_clusters]
        centroids = samples[centroids_idx]

        # EM-style alternative minimization
        prev_error = None
        for n_iter in range(max_iters):
            # Computer cluster assignments
            l2_dist = torch.cdist(samples, centroids, p=2.0)
            assignment = l2_dist.argmin(dim=-1)

            # Update cluster centroids
            centroids = torch.vstack(
                [
                    samples[assignment == cluster_id].mean(dim=0)
                    for cluster_id in range(n_clusters)
                ]
            )

            # K-means loss: sum of squared L2 distances between
            # each point and its assigned cluster.
            error = l2_dist.min(dim=-1).values.pow(2.0).sum()
            if prev_error is not None:
                if (prev_error - error).abs() / prev_error < rtol:
                    break
            prev_error = error

        log.info(
            f"K-means with {n_samples} samples and {n_clusters} cluster "
            f"converged after {n_iter + 1} iters."
        )

        return centroids
