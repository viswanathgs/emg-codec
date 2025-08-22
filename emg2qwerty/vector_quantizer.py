import abc

import logging
import math

import torch
from torch import nn

from emg2qwerty import utils

log = logging.getLogger(__name__)

EPS = 1e-8


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
        ema_update: bool = True,
        ema_decay: float = 0.9,
        ema_expire_code_threshold: float = 2.0,
    ) -> None:
        super().__init__(
            parent_block_id=parent_block_id,
            embedding_dim=embedding_dim,
        )

        self.codebook_size = codebook_size
        self.kmeans_init = kmeans_init
        self.kmeans_max_iters = kmeans_max_iters
        self.kmeans_rtol = kmeans_rtol
        self.ema_update = ema_update
        self.ema_decay = ema_decay
        self.ema_expire_code_threshold = ema_expire_code_threshold

        # Init codebook entries randomly. Will be re-initialized
        # using k-means on the first training batch if kmeans_init is True.
        self.codebook = nn.Embedding(codebook_size, embedding_dim)

        # Whether the codebook has been initialized
        self.register_buffer("codebook_initialized", torch.tensor(False))

        if self.ema_update:
            # Disable gradient updates to the codebook if using EMA updates
            # (as opposed to codebook or commitment loss).
            self.codebook.requires_grad_(False)

            # Buffers maintaining EMA cluster sums and counts for codebook updates
            self.register_buffer(
                "ema_cluster_sum",
                torch.zeros(codebook_size, embedding_dim, dtype=torch.float32),
            )
            self.register_buffer(
                "ema_cluster_size",
                torch.zeros(codebook_size, dtype=torch.float32),
            )

            # Whether the EMA buffers have been initialized
            self.register_buffer("ema_initialized", torch.tensor(False))

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        # inputs: TNC
        assert inputs.ndim == 3
        assert inputs.shape[2] == self.embedding_dim

        # Lazy-init codebook entries based on the first training batch.
        # No-op if kmeans_init is False.
        if self.training and not self.codebook_initialized:
            self._init_codebook(inputs)
            self.codebook_initialized.fill_(True)

        l2_dist = torch.cdist(inputs, self.codebook.weight, p=2.0)
        assignment = l2_dist.argmin(dim=-1)
        quantized = self.codebook(assignment)

        # Straight-through estimator (STE) to bypass the non-differentiable
        # codebook lookup operation.
        quantized = (quantized - inputs).detach() + inputs

        if self.training and self.ema_update:
            self._update_codebook(inputs, assignment)
            self._expire_codes(inputs)

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
    def _update_codebook(self, inputs: torch.Tensor, assignment: torch.Tensor) -> None:
        # inputs: TNC, assignment: TN
        assert inputs.ndim == assignment.ndim + 1
        assert inputs.shape[-1] == self.embedding_dim

        # Compute cluster sums and counts for this batch
        cluster_sum = torch.vstack(
            [
                inputs[assignment == cluster_id].sum(dim=0)
                for cluster_id in range(self.codebook_size)
            ]
        )
        cluster_size = torch.tensor(
            [
                (assignment == cluster_id).sum()
                for cluster_id in range(self.codebook_size)
            ],
            dtype=inputs.dtype,
            device=inputs.device,
        )

        # All-reduce cluster sums and counts across all ranks
        utils.all_reduce_tensor(cluster_sum)
        utils.all_reduce_tensor(cluster_size)

        # Initialize EMA buffers using the first training batch
        decay = self.ema_decay
        if not self.ema_initialized:
            decay = 0.0
            self.ema_initialized.fill_(True)

        # Update EMA cluster sums and counts:
        # ema_cluster_sum = decay * ema_cluster_sum + (1 - decay) * cluster_sum
        # ema_cluster_size = decay * ema_cluster_size + (1 - decay) * cluster_size
        self.ema_cluster_sum.mul_(decay).add_(cluster_sum, alpha=1 - decay)
        self.ema_cluster_size.mul_(decay).add_(cluster_size, alpha=1 - decay)

        # Update codebook entries using EMA
        new_centroids = self.ema_cluster_sum / (
            self.ema_cluster_size.unsqueeze(1) + EPS
        )
        self.codebook.weight.copy_(new_centroids)

    @torch.jit.ignore
    def _expire_codes(self, inputs: torch.Tensor) -> None:
        # Find indices of codes where the EMA cluster size falls below threshold
        expired_idx = torch.where(
            self.ema_cluster_size < self.ema_expire_code_threshold
        )[0]

        n_expired = len(expired_idx)
        if len(expired_idx) == 0:
            # All codes are sufficiently active, nothing to expire
            return

        log.info(
            f"Found {n_expired}/{self.codebook_size} expired codes; "
            f"replacing them with random samples from current batch in rank 0."
        )

        # Replace expired codes with samples randomly selected from the current
        # training batch. Perform this just on rank 0 and then broadcast to keep the
        # codebook in sync across ranks.
        if utils.get_rank() == 0:
            # Flatten input batch to select replacements from
            samples = inputs.view(-1, self.embedding_dim)
            n_samples = len(samples)
            if n_samples < n_expired:
                log.warning(
                    "Not enough samples in rank 0 training batch "
                    "to replace expired codes."
                )
                expired_idx = expired_idx[:n_samples]
                n_expired = n_samples

            # Sample replacements and update expired codes in the codebook
            replacement_idx = torch.randperm(n_samples)[:n_expired]
            self.codebook.weight[expired_idx] = samples[replacement_idx]

            # If EMA buffers are present, update them for the expired codes
            if self.ema_update:
                self.ema_cluster_sum[expired_idx] = samples[replacement_idx]
                self.ema_cluster_size[expired_idx] = 1.0

        # Broadcast the updated codebook to all ranks
        utils.broadcast_tensor(self.codebook.weight, src_rank=0)
        if self.ema_update:
            utils.broadcast_tensor(self.ema_cluster_sum, src_rank=0)
            utils.broadcast_tensor(self.ema_cluster_size, src_rank=0)

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


class ResidualVectorQuantizer(BaseVectorQuantizer):
    def __init__(
        self,
        parent_block_id: int,
        embedding_dim: int,
        n_vq: int,
        codebook_size: int,
        kmeans_init: bool = True,
        kmeans_max_iters: int = 300,
        kmeans_rtol: float = 1e-6,
        ema_update: bool = True,
        ema_decay: float = 0.9,
        ema_expire_code_threshold: float = 2.0,
    ) -> None:
        super().__init__(
            parent_block_id=parent_block_id,
            embedding_dim=embedding_dim,
        )

        self.vq = nn.ModuleList(
            VectorQuantizer(
                parent_block_id=parent_block_id,
                embedding_dim=embedding_dim,
                codebook_size=codebook_size,
                kmeans_init=kmeans_init,
                kmeans_max_iters=kmeans_max_iters,
                kmeans_rtol=kmeans_rtol,
                ema_update=ema_update,
                ema_decay=ema_decay,
                ema_expire_code_threshold=ema_expire_code_threshold,
            )
            for _ in range(n_vq)
        )
        log.info(f"Instantiated ResidualVectorQuantizer with {n_vq} levels.")

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        quantized = torch.zeros_like(inputs)
        for vq in self.vq:
            # Quantize the residual
            quantized += vq(inputs - quantized)
        return quantized

    @torch.jit.export
    def nbits_compressed(self) -> int:
        return sum([vq.nbits_compressed() for vq in self.vq])
