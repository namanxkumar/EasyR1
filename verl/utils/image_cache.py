"""Disk-based cache for multi-modal tensor data.

Saves pre-computed pixel_values / image_grid_thw tensors to disk so they can
be evicted from memory after rollout and lazily loaded at training time.
"""

from __future__ import annotations

import logging
import os
import shutil
import uuid

import torch

logger = logging.getLogger(__name__)

# Sentinel key used to detect cached-to-disk multi_modal_data dicts
CACHE_PATH_KEY = "__image_cache_path__"


def save_multi_modal_data(
    multi_modal_data: dict,
    cache_dir: str,
) -> dict:
    """Save multi-modal tensors to disk and return a placeholder dict.

    Args:
        multi_modal_data: dict with tensor values (pixel_values, image_grid_thw, etc.)
        cache_dir: directory to write .pt files into

    Returns:
        A small dict ``{CACHE_PATH_KEY: "/path/to/file.pt"}`` that replaces
        the heavy tensor dict in the DataProto.
    """
    if not multi_modal_data:
        return multi_modal_data

    os.makedirs(cache_dir, exist_ok=True)
    filename = f"{uuid.uuid4().hex}.pt"
    path = os.path.join(cache_dir, filename)
    torch.save(multi_modal_data, path)
    return {CACHE_PATH_KEY: path}


def load_multi_modal_data(placeholder: dict) -> dict:
    """Load multi-modal tensors from a cache placeholder.

    Args:
        placeholder: dict containing ``CACHE_PATH_KEY`` pointing to a .pt file

    Returns:
        The original multi-modal data dict with tensors.
    """
    path = placeholder[CACHE_PATH_KEY]
    data = torch.load(path, map_location="cpu", weights_only=True)
    return data


def cleanup_cache_dir(cache_dir: str) -> None:
    """Remove all files in the cache directory."""
    if cache_dir and os.path.isdir(cache_dir):
        n_files = len(os.listdir(cache_dir))
        shutil.rmtree(cache_dir, ignore_errors=True)
        logger.info(f"Cleaned up image cache: removed {n_files} files from {cache_dir}")
