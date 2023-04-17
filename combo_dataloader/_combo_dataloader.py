"""
Dataloader that uses a producer/consumer model to parallelize multiple dataloaders.
"""

from typing import List, Dict, Any, Optional, Callable
from collections import namedtuple
from dataclasses import dataclass
import enum
import threading
import atexit
from torch import multiprocessing
from ._dataloader import DataLoader, DataLoaderParams
from ._transform import ComboDLTransform
from ._dali_dataloader import DaliDataLoader
from ._pytorch_dataloader import PytorchDataloader


# Maximum size of producer/consumer queue of batches
MAX_QUEUE_SIZE = 50


class DataLoaderType(enum.Enum):
    # value doesn't matter
    DALI = enum.auto()
    PYTORCH = enum.auto()


# Events that can be sent to the dataloader run process
@dataclass
class ShutdownEvent():
    """
    The subthread populating the queue with batches should stop, without restarting
    """
    pass

@dataclass 
class StartEvent():
    """
    Start a thread to populate the given `batch_queue`with batches, after stopping
    the current subthread if one is running
    """
    pass

def run_dataloader(
        dataloader: DataLoaderType,
        params: DataLoaderParams,
        event_queue,
    ) -> None:
        # On restart event, restart iteration for dataloader
        thread = None
        shutdown_communicator = threading.Event()

        # Event queue
        event = event_queue.get()
        while event is not ShutdownEvent:
            if event is StartEvent:
                # end current subthread
                if thread:
                    shutdown_communicator.set()
                    thread.join()

                # start new subthread
                thread = threading.Thread(
                    target=populate_queue, name=dataloader.name, args=(
                    dataloader, params, event.batch_queue, shutdown_communicator,
                ))

                # restart iteration
                thread.start()

            event = event_queue.get()

        # clean up subthread
        if thread:
            shutdown_communicator.set()
            thread.join()


def populate_queue(
        dataloader: DataLoaderType,
        params: DataLoaderParams,
        queue,
        break_iteration_event,
    ) -> None:
    """
    Load batches of data and add them to the queue
    """

    clip_count = 0

    if len(params.video_paths) > 0:
        # TODO: is it possible to avoid re-construction?
        if dataloader == DataLoaderType.DALI:
            dl = DaliDataLoader(params)
        elif dataloader == DataLoaderType.PYTORCH:
            dl = PytorchDataloader(params)

        for batch in dl:
            if break_iteration_event.is_set():
                break

            clip_count += len(batch["frames"])

            queue.put(batch)

    queue.put("done")


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
        self._ctx = ctx
        self._event_queue = ctx.Queue()
        self._lock = ctx.Lock()
        self._lock.acquire()

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
        # Send start event to each process to start (or restart) queue population
        for event_queue in self._event_queues:
            event_queue.put(StartEvent(batch_queue))

        # Instantiate new iterator
        return _ComboDataloaderIterator(self._dataloaders, self._params, self._ctx)

def _ComboDataloaderIterator():
    def __init__(self, dataloader_types, dataloader_params, multiprocessing_context):
        self._batch_queue = multiprocessing_context.Queue(MAX_QUEUE_SIZE) # stores Dict[str, Tensor]
        self._done_count = len(dataloader_types)

        # Create separate event queues for each process to ensure delivery
        # of each event to each dataloader
        self._event_queues = [multiprocessing_context.Queue() for dl in dataloader_types]

        # Create processes for each dataloader
        self._dl_processes = [
            multiprocessing_context.Process(
                target=run_dataloader,
                args=(
                    dataloader,
                    params,
                    event_queue,
                ),
                daemon=(dataloader == DataLoaderType.DALI)
            )
            for dataloader, params, event_queue in zip(dataloader_types, dataloader_params, self._event_queues)
        ]

        for queue in self._event_queues:
            queue.put(StartEvent())


    def __next__(self):
        """
        Return the next batch of inputs. Tensor input is stored under the "frames"
        key of the result dictionary; labels under the "label" key.
        NOTE: may add labels to the output in the future.
        """
        # Start the dataloader processes if not already
        # They will not start populating the queue until a "start" event is sent
        if not self._started:
            self._started = True
            for t in self._dl_processes:
                t.start()


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
        # Send shutdown event to each process
        for event_queue in self._event_queues:
            event_queue.put(ShutdownEvent())


