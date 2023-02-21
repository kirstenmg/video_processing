"""
Dataloader that uses a producer/consumer model to parallelize multiple dataloaders.
"""

from typing import List, Dict, Any, Optional, Callable
from collections import namedtuple
from torch import multiprocessing
import enum
import atexit
from .dataloader import DataLoader, DataLoaderParams
from .transform import ComboDLTransform
from .dali_dataloader import DaliDataLoader
from .pytorch_dataloader import PytorchDataloader


# Maximum size of producer/consumer queue of batches
MAX_QUEUE_SIZE = 50


class DataLoaderType(enum.Enum):
    # value doesn't matter
    DALI = enum.auto()
    PYTORCH = enum.auto()


def populate_queue(
        dataloader: DataLoaderType,
        queue,
        lock,
        shutdown_event,
        params: DataLoaderParams,
    ) -> None:
    """
    Load batches of data and add them to the queue
    """

    clip_count = 0

    if len(params.video_paths) > 0:
        if dataloader == DataLoaderType.DALI:
            dl = DaliDataLoader(params)
        elif dataloader == DataLoaderType.PYTORCH:
            dl = PytorchDataloader(params)

        for batch in dl:
            if shutdown_event.is_set():
                break

            clip_count += len(batch["frames"])

            queue.put(batch)

    queue.put("done")

    # Block until shutdown (hacky)
    lock.acquire()
    lock.release()


class ComboDataLoader(DataLoader):
    def __init__(
        self,
        dataloaders: List[DataLoaderType],
        dataloader_portions: List[int],
        video_paths: List[str],
        labels: Optional[List[int]] = None,
        *,
        sequence_length: int,
        fps: int,
        transform: Optional[ComboDLTransform] = None,
        stride: int = 1,
        step: int = -1, # defaults to sequence_length
        batch_size: int = 1,
        pytorch_dataloader_kwargs: Optional[Dict[str, Any]] = None,
        pytorch_dataset_kwargs: Optional[Dict[str, Any]] = None,
        pytorch_additional_transform: Optional[Callable] = None,
        dali_pipeline_kwargs: Optional[Dict[str, Any]] = None,
        dali_reader_kwargs:Optional[Dict[str, Any]] = None,
        dali_additional_transform: Optional[Callable] = None,
    ):
        """
        TODO: docs
        """

        if len(dataloaders) != len(dataloader_portions) or len([num for num in dataloader_portions if num < 0]) > 0 or sum(dataloader_portions) == 0:
            raise ValueError(f'Dataloader portions (count {len(dataloader_portions)})' +\
            f' must be positive and map 1:1 to dataloaders (count {len(dataloader_portions)})')

        if labels and len(labels) != len(video_paths):
            raise ValueError(f'Number of labels ({len(labels)}) must equal number of videos ({len(video_paths)})')

        if not pytorch_dataloader_kwargs:
            pytorch_dataloader_kwargs = dict()

        if not pytorch_dataset_kwargs:
            pytorch_dataset_kwargs = dict()

        if not dali_pipeline_kwargs:
            dali_pipeline_kwargs = dict()

        if not dali_reader_kwargs:
            dali_reader_kwargs = dict()

        
        # Still shut down in case of early termination
        atexit.register(self.shutdown)

        # For lazy starting of dataloader processes
        self._started = False

        # Split up video paths
        StartEnd = namedtuple("StartEnd", ["start", "end"])

        portion_sum = sum(dataloader_portions)
        video_ranges = []
        start = 0
        for i in range(len(dataloaders)):
            end = int(round(start + (dataloader_portions[i] / portion_sum) * len(video_paths)))
            video_ranges.append(StartEnd(start, end))
            start = end

        video_ranges[-1] = StartEnd(video_ranges[-1].start, len(video_paths))

        ctx = multiprocessing.get_context('spawn')
        self._batch_queue = ctx.Queue(MAX_QUEUE_SIZE)  # stores Dict[str, Tensor]
        self._lock = ctx.Lock()
        self._lock.acquire()
        self._shutdown_event = ctx.Event()

        self._dl_processes = [
            ctx.Process(
                target=populate_queue,
                args=(
                    dataloader, 
                    self._batch_queue,
                    self._lock,
                    self._shutdown_event,
                    DataLoaderParams(
                        video_paths=video_paths,
                        sequence_length=sequence_length,
                        fps=fps,
                        stride=stride,
                        step=step,
                        batch_size=batch_size,
                        labels=labels,
                        transform=transform,
                        pytorch_dataloader_kwargs=pytorch_dataloader_kwargs,
                        pytorch_dataset_kwargs=pytorch_dataset_kwargs,
                        pytorch_additional_transform=pytorch_additional_transform,
                        dali_pipeline_kwargs=dali_pipeline_kwargs,
                        dali_reader_kwargs=dali_reader_kwargs,
                        dali_additional_transform=dali_additional_transform,
                    )
                ),
                daemon=(dataloader == DataLoaderType.DALI)
            )
            for video_range, dataloader in zip(video_ranges, dataloaders)
        ]

        self._done_count = len(dataloaders)

    def __next__(self):
        """
        Return the next batch of inputs. Tensor input is stored under the "video"
        key of the result dictionary.
        NOTE: may add labels to the output in the future.
        """

        if not self._started:
            for t in self._dl_processes:
                t.start()
            self._started = True

        # Return next item in queue, blocking
        next_item = self._batch_queue.get();

        while next_item == "done":
            # Decrement count and check if all sub-dataloaders are done
            self._done_count -= 1
            if self._done_count == 0:
                assert self._batch_queue.empty()
                raise StopIteration
            else:
                next_item = self._batch_queue.get()

        return next_item

    def shutdown(self):
        if not self._shutdown_event.is_set():
            self._shutdown_event.set()
            self._lock.release()

