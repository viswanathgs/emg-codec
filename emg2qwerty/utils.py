# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import logging
import os
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import torch

import torch.distributed as dist
from hydra.core.hydra_config import HydraConfig

from hydra.utils import get_original_cwd, instantiate
from omegaconf import DictConfig, OmegaConf
from torch import nn

log = logging.getLogger(__name__)


def instantiate_optimizer_and_scheduler(
    params: Iterator[nn.Parameter],
    optimizer_config: DictConfig,
    lr_scheduler_config: DictConfig,
) -> dict[str, Any]:
    optimizer = instantiate(optimizer_config, params)
    scheduler = instantiate(lr_scheduler_config.scheduler, optimizer)
    lr_scheduler = instantiate(lr_scheduler_config, scheduler=scheduler)
    return {
        "optimizer": optimizer,
        "lr_scheduler": OmegaConf.to_container(lr_scheduler),
    }


def get_last_checkpoint(checkpoint_dir: Path) -> Path | None:
    checkpoints = list(checkpoint_dir.glob("*.ckpt"))
    if not checkpoints:
        return None
    return max(checkpoints, key=lambda p: p.stat().st_mtime)


def cpus_per_task(gpus_per_node: int, tasks_per_node: int, num_workers: int) -> int:
    """Number of CPUs to request per task per node taking into account
    the number of GPUs and dataloading workers."""
    gpus_per_task = gpus_per_node // tasks_per_node
    if gpus_per_task <= 0:
        return num_workers + 1
    else:
        return (num_workers + 1) * gpus_per_task


def get_rank() -> int:
    if dist.is_available() and dist.is_initialized():
        return int(dist.get_rank())
    return 0


def broadcast_tensor(tensor: torch.Tensor, src_rank: int) -> None:
    if dist.is_available() and dist.is_initialized():
        dist.broadcast(tensor, src=src_rank)


def all_reduce_tensor(
    tensor: torch.Tensor, op: dist.ReduceOp = dist.ReduceOp.SUM
) -> None:
    if dist.is_available() and dist.is_initialized():
        dist.all_reduce(tensor, op=op)


def set_python_path() -> None:
    """Add working dir to PYTHONPATH."""
    working_dir = get_original_cwd()
    python_paths = os.environ.get("PYTHONPATH", "").split(os.pathsep)
    if working_dir not in python_paths:
        python_paths.append(working_dir)
        os.environ["PYTHONPATH"] = os.pathsep.join(python_paths)


def set_cuda_visible_devices_for_sweep(config: DictConfig) -> None:
    """Set CUDA_VISIBLE_DEVICES from hydra job number for sweep runs."""
    # Only applicable if using a single GPU per job in a sweep
    if config.trainer.devices != 1:
        return

    # Get the serial number of job within a sweep from hydra
    try:
        hydra_cfg = HydraConfig().get()
        job_num = hydra_cfg.job.get("num")
    except Exception:
        job_num = None

    # Set CUDA_VISIBLE_DEVICES based on job number within sweep
    if job_num is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(job_num)
        log.info(f"Set CUDA_VISIBLE_DEVICES={job_num} from hydra.job.num in sweep")
