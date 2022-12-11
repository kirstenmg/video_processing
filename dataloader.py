"""
Interface for a dataloader, which takes in video data, decodes and transforms
the data, and provides the pixel input in batches to a model.
"""

from abc import ABC, abstractmethod
from typing import Dict
from torch import Tensor

class DataLoader(ABC):
    def __iter__(self):
        return self

