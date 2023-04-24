"""
Dataloader that uses a producer/consumer model to parallelize multiple dataloaders.
"""

from typing import List, Dict, Any, Optional, Callable
from collections import namedtuple
from dataclasses import dataclass
import enum
import atexit
from torch import multiprocessing
from ._dataloader import DataLoader, DataLoaderParams
from ._transform import ComboDLTransform
from ._dali_dataloader import DaliDataLoader
from ._pytorch_dataloader import PytorchDataLoader


# Maximum size of producer/consumer queue of batches
MAX_QUEUE_SIZE = 50


class DataLoaderType(enum.Enum):
    # value doesn't matter
    DALI = enum.auto()
    PYTORCH = enum.auto()


# Events that can be sent to the dataloader run process
START_EVENT = "start"
SHUTDOWN_EVENT = "shutdown"

# Indicator that worker is done processing all batches
DONE = "done"

def run_dataloader(
        dataloader: DataLoaderType,
        params: DataLoaderParams,
        batch_queue,
        event_queue,
        break_iteration_event,
    ) -> None:
        if len(params.video_paths) <= 0:
            return

        if dataloader == DataLoaderType.DALI:
            dl = DaliDataLoader(params)
        elif dataloader == DataLoaderType.PYTORCH:
            dl = PytorchDataLoader(params)

        # TODO: what if populators are never started?
        event = event_queue.get()
        while event != SHUTDOWN_EVENT:
            if event == START_EVENT:
                populate_queue(dl, batch_queue, break_iteration_event)

            event = event_queue.get()


def populate_queue(
        dataloader,
        queue,
        break_iteration_event,
    ) -> None:
    """
    Load batches of data and add them to the queue
    """

    for batch in dataloader:
        if break_iteration_event.is_set():
            break
        
        queue.put(batch)

    queue.put(DONE)


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
        Constructs a combined dataloader.

        Arguments:
        dataloaders: a list of dataloader types to create subprocesses for
        dataloader_portions: list of integers representing the portion of videos
        to allocate to each dataloader; length must be equal to `dataloaders` length
        video_paths: paths of the videos to load
        labels: labels associated with videos in video_paths, must have same length
        as video_paths if provided
        transform: transform to apply to each clip
        stride: distance between consecutive frames in the clip
        step: frame interval between each clip
        sequence_length: frames to load per clip
        batch_size: the number of clips returned in a batch
        pytorch_dataloader_kwargs: keyword arguments to pass to torch Dataloader constructor
        pytorch_dataset_kwargs: keyword arguments to pass to LabeledVideoDataset constructor
        dali_pipeline_kwargs: keyword arguments to pass to DALI pipeline_def function call
        dali_reader_kwargs: keyword arguments to pass to fn.readers.video_resize
        """

        if len(dataloaders) != len(dataloader_portions) or len([num for num in dataloader_portions if num < 0]) > 0 or sum(dataloader_portions) == 0:
            raise ValueError(f'Dataloader portions (count {len(dataloader_portions)}, sum {sum(dataloader_portions)})' +\
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
        self._ctx = ctx
        self._event_queue = ctx.Queue()
        self._lock = ctx.Lock()
        self._lock.acquire()

        # Set up dataloaders and params for iterator
        self._iterator = None
        self._dataloaders = dataloaders
        self._params = [
            DataLoaderParams(
                video_paths=video_paths[video_range.start:video_range.end],
                sequence_length=sequence_length,
                fps=fps,
                stride=stride,
                step=step if step >= 0 else sequence_length,
                batch_size=batch_size,
                labels=labels if not labels else labels[video_range.start:video_range.end],
                transform=transform if transform else ComboDLTransform(),
                pytorch_dataloader_kwargs=pytorch_dataloader_kwargs,
                pytorch_dataset_kwargs=pytorch_dataset_kwargs,
                pytorch_additional_transform=pytorch_additional_transform,
                dali_pipeline_kwargs=dali_pipeline_kwargs,
                dali_reader_kwargs=dali_reader_kwargs,
                dali_additional_transform=dali_additional_transform,
            )
            for video_range in video_ranges
        ]


    def __iter__(self):
        if not self._iterator:
            # Instantiate new iterator
            self._iterator = _ComboDataLoaderIterator(self._dataloaders, self._params, self._ctx)
        else:
            # Reuse existing iterator
            self._iterator._restart()

        return self._iterator

class _ComboDataLoaderIterator():
    def __init__(self, dataloader_types, dataloader_params, multiprocessing_context):
        self._batch_queue = multiprocessing_context.Queue(MAX_QUEUE_SIZE) # stores Dict[str, Tensor]
        self._unfinished_count = len(dataloader_types)

        # Create separate event queues for each process to ensure delivery
        # of each event to each dataloader
        self._event_queues = [multiprocessing_context.Queue() for dl in dataloader_types]
        self._break_iteration_event = multiprocessing_context.Event();
        self._break_iteration_event.clear()

        # Create processes for each dataloader
        self._dl_processes = [
            multiprocessing_context.Process(
                target=run_dataloader,
                args=(
                    dataloader,
                    params,
                    self._batch_queue,
                    event_queue,
                    self._break_iteration_event,
                ),
                daemon=(dataloader == DataLoaderType.DALI)
            )
            for dataloader, params, event_queue in zip(dataloader_types, dataloader_params, self._event_queues)
        ]

        # Still shutdown in case of early exit
        atexit.register(self.shutdown)

        for t in self._dl_processes:
            t.start()

        # TODO: could also do this lazily, in __next__. Consult torch for ref.
        # Restart subprocess iteration
        for event_queue in self._event_queues:
            event_queue.put(START_EVENT)

    def __iter__(self):
        return self

    def __del__(self):
        self.shutdown()

    def _restart(self):
        # Break iteration of unfinished subprocesses and wait for them to finish
        if self._unfinished_count:
            self._break_iteration_event.set()

            while self._unfinished_count:
                next_item = self._batch_queue.get()
                if next_item == "done":
                    self._unfinished_count -= 1

            self._break_iteration_event.clear()

        self._unfinished_count = len(self._dl_processes)

        # Restart subprocess iteration
        for event_queue in self._event_queues:
            event_queue.put(START_EVENT)

    def __next__(self):
        """
        Return the next batch of inputs. Tensor input is stored under the "frames"
        key of the result dictionary; labels under the "label" key.
        """

        # Return next item in queue, blocking
        next_item = self._batch_queue.get();

        while next_item == DONE:
            # Decrement count and check if all sub-dataloaders are done
            self._unfinished_count -= 1
            if self._unfinished_count == 0:
                assert self._batch_queue.empty()
                raise StopIteration
            else:
                next_item = self._batch_queue.get()

        return next_item

    def shutdown(self):
        # Send shutdown event to each process
        self._break_iteration_event.set()
        for event_queue in self._event_queues:
            event_queue.put(SHUTDOWN_EVENT)


