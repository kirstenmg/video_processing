"""
Dataloader that uses a producer/consumer model to parallelize multiple dataloaders.
"""

from typing import List
from collections import namedtuple
from dataloader import DataLoader
from dali_dataloader import DaliDataLoader
from pytorch_dataloader import PytorchDataloader
from torch import multiprocessing, cuda
import enum
import atexit


class DataLoaderType(enum.Enum):
    # value doesn't matter
    DALI = enum.auto()
    PYTORCH = enum.auto()


def populate_queue(
        dataloader: DataLoaderType,
        video_paths: List[str],
        queue,
        lock,
        shutdown_event,
        
    ) -> None:

    if dataloader == DataLoaderType.DALI:
        dl = DaliDataLoader(8, 10, video_paths)
    elif dataloader == DataLoaderType.PYTORCH:
        dl = PytorchDataloader(8, 10, video_paths)

    for batch in dl:
        if shutdown_event.is_set():
            break
        queue.put(batch)

    queue.put("done")

    # Block until shutdown
    # If some error occurs in the main process, I think this will block infinitely
    # TODO: add atexit.shutdown, maybe this will solve the issue
    #if dataloader == DataLoaderType.DALI:
    lock.acquire()
    lock.release()


class ComboDataLoader(DataLoader):
    def __init__(self, dataloaders: List[DataLoaderType], video_paths: List[str]):
        # Still shut down in case of early termination
        atexit.register(self.shutdown)

        self._started = False

        # Split up video paths
        vids_per_dl = len(video_paths) // len(dataloaders)
        StartEnd = namedtuple("StartEnd", ["start", "end"])
        video_ranges = [
            StartEnd(i * vids_per_dl, (i + 1) * vids_per_dl)
            for i in range(len(dataloaders))
        ]
        video_ranges[-1] = StartEnd(video_ranges[-1].start, len(video_paths))

        ctx = multiprocessing.get_context('spawn')
        self._batch_queue = ctx.Queue()  # stores Dict[str, Tensor]
        self._lock = ctx.Lock()
        self._lock.acquire()
        self._shutdown_event = ctx.Event()

        self._dl_processes = [
            ctx.Process(
                target=populate_queue,
                args=(
                    dataloader, 
                    video_paths[video_range.start:video_range.end], 
                    self._batch_queue,
                    self._lock,
                    self._shutdown_event,
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

    def __del__(self):
        print("deletion")

    def shutdown(self):
        print("in here")
        self._shutdown_event.set()
        self._lock.release()

