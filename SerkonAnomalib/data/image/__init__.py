"""Anomalib Image Datasets.

This module contains the supported image datasets for Anomalib.
"""

# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0


from enum import Enum


from .folder import Folder



class ImageDataFormat(str, Enum):
    """Supported Image Dataset Types."""

    FOLDER = "folder"
    FOLDER_3D = "folder_3d"


__all__ = ["Folder"]
