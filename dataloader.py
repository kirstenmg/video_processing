"""
Interface for a dataloader, which takes in video data, decodes and transforms
the data, and provides the pixel input in batches to a model.
"""

from abc import ABC, abstractmethod
from typing import Dict
from torch import Tensor

class DataLoader(ABC):
    @abstractmethod
    def next_batch(self) -> Dict[str, Tensor]:
        """
        Return the next batch of inputs, or None if no more input.
        Tensor input is stored under the "frames" key of the result dictionary.
        NOTE: may add labels to the output in the future.
        """
        pass

    def __iter__(self):
        return self

    def __next__(self):
        return self.next_batch()

