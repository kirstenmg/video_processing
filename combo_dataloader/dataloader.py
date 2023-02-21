"""
Interface for a dataloader, which takes in video data, decodes and transforms
the data, and provides the pixel input in batches to a model.
"""

from abc import ABC
from dataclasses import dataclass
from typing import Dict, List, Any, Callable
from .transform import ComboDLTransform

@dataclass
class DataLoaderParams:
    video_paths: List[str]
    sequence_length: int
    fps: int
    stride: int
    step: int
    batch_size: int
    labels: List[int]
    transform: ComboDLTransform
    pytorch_dataloader_kwargs: Dict[str, Any]
    pytorch_dataset_kwargs: Dict[str, Any]
    pytorch_additional_transform: Callable
    dali_pipeline_kwargs: Dict[str, Any]
    dali_reader_kwargs: Dict[str, Any]
    dali_additional_transform: Callable

class DataLoader(ABC):
    def __iter__(self):
        return self

