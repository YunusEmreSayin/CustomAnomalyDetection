"""Anomalib Datasets."""

# Copyright (C) 2022-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import importlib
import logging
from enum import Enum
from itertools import chain

from omegaconf import DictConfig, ListConfig

from SerkonAnomalib.utils.config import to_tuple

from .base import AnomalibDataModule, AnomalibDataset
from .depth import DepthDataFormat, Folder3D
from .image import Folder, ImageDataFormat
from .predict import PredictDataset
from .utils import LabelName

logger = logging.getLogger(__name__)


DataFormat = Enum(  # type: ignore[misc]
    "DataFormat",
    {i.name: i.value for i in chain(DepthDataFormat, ImageDataFormat)},
)


class UnknownDatamoduleError(ModuleNotFoundError):
    ...


def get_datamodule(config: DictConfig | ListConfig | dict) -> AnomalibDataModule:
    """Get Anomaly Datamodule.

    Args:
        config (DictConfig | ListConfig | dict): Configuration of the anomaly model.

    Returns:
        PyTorch Lightning DataModule
    """
    logger.info("Loading the datamodule")

    if isinstance(config, dict):
        config = DictConfig(config)

    try:
        _config = config.data if "data" in config else config
        if len(_config.class_path.split(".")) > 1:
            module = importlib.import_module(".".join(_config.class_path.split(".")[:-1]))
        else:
            module = importlib.import_module("anomalib.data")
    except ModuleNotFoundError as exception:
        logger.exception(f"ModuleNotFoundError: {_config.class_path}")
        raise UnknownDatamoduleError from exception
    dataclass = getattr(module, _config.class_path.split(".")[-1])
    init_args = {**_config.get("init_args", {})}  # get dict
    if "image_size" in init_args:
        init_args["image_size"] = to_tuple(init_args["image_size"])

    return dataclass(**init_args)


__all__ = [
    "AnomalibDataset",
    "AnomalibDataModule",
    "DepthDataFormat",
    "ImageDataFormat",
    "get_datamodule",

    "Folder",
    "Folder3D",
    "PredictDataset",

    "LabelName",
]
