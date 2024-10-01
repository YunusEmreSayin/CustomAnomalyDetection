from enum import Enum

from .folder_3d import Folder3D


class DepthDataFormat(str, Enum):
    """Supported Depth Dataset Types."""

    FOLDER_3D = "folder_3d"


__all__ = ["Folder3D"]