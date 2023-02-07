"""
Dataloader that uses a producer/consumer model to parallelize multiple dataloaders.
"""

import os # to get pid for benchmarking
from typing import List
from collections import namedtuple
from dataloader import DataLoader
from dali_dataloader import DaliDataLoader
from pytorch_dataloader import PytorchDataloader
from torch import multiprocessing, cuda
import multiprocessing as mp
import enum
import atexit
import datetime
import time
import duckdb_wrapper
import torch


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
        benchmark_results_queue: mp.Queue,
    ) -> None:
    """
    Load batches of data and add them to the queue
    """

    clip_count = 0

    if len(video_paths) > 0:
        if dataloader == DataLoaderType.DALI:
            dl = DaliDataLoader(8, 10, video_paths)
        elif dataloader == DataLoaderType.PYTORCH:
            dl = PytorchDataloader(8, 10, video_paths)

        for batch in dl:
            if shutdown_event.is_set():
                break

            clip_count += len(batch["frames"])

            size_before = queue.qsize()
            start = datetime.datetime.now()
            perf_start = time.perf_counter()

            queue.put(batch)

            duration = time.perf_counter() - perf_start
            size_after = queue.qsize()

            benchmark_results_queue.put(duckdb_wrapper.QueueBlockEntry(
                "put",
                os.getpid(),
                dataloader.name,
                start,
                size_before,
                size_after,
                duration,
            ))

    queue.put("done")

    print(f'{dataloader.name} processed {clip_count} clips')

    # Block until shutdown (hacky)
    lock.acquire()
    lock.release()


class ComboDataLoader(DataLoader):
    def __init__(
        self,
        dataloaders: List[DataLoaderType],
        video_paths: List[str],
        dataloader_portions: List[int],
        queue_size: int,
        benchmark_results_queue: multiprocessing.Queue
    ):
        if len(dataloaders) != len(dataloader_portions) or len([num for num in dataloader_portions if num < 0]) > 0 or sum(dataloader_portions) == 0:
            raise ValueError(f'Dataloader portions (count {len(dataloader_portions)})' +\
            f' must be positive and map 1:1 to dataloaders (count {len(dataloader_portions)})')

        # Still shut down in case of early termination
        atexit.register(self.shutdown)

        # For lazy starting of dataloader processes
        self._started = False

        self._benchmark_results_queue = benchmark_results_queue

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
        self._batch_queue = ctx.Queue(queue_size)  # stores Dict[str, Tensor]
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
                    benchmark_results_queue,
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
        size_before = self._batch_queue.qsize()
        start = datetime.datetime.now()
        perf_start = time.perf_counter()

        next_item = self._batch_queue.get();

        duration = time.perf_counter() - perf_start
        size_after = self._batch_queue.qsize()

        self._benchmark_results_queue.put(duckdb_wrapper.QueueBlockEntry(
            "get",
            os.getpid(),
            "NONE",
            start,
            size_before,
            size_after,
            duration,
        ))


        while next_item == "done":
            # Decrement count and check if all sub-dataloaders are done
            self._done_count -= 1
            if self._done_count == 0:
                assert self._batch_queue.empty()
                raise StopIteration
            else:
                size_before = self._batch_queue.qsize()
                start = datetime.datetime.now()
                perf_start = time.perf_counter()
                
                next_item = self._batch_queue.get()

                duration = time.perf_counter() - perf_start
                size_after = self._batch_queue.qsize()

                self._benchmark_results_queue.put(duckdb_wrapper.QueueBlockEntry(
                    "get",
                    os.getpid(),
                    "NONE",
                    start,
                    size_before,
                    size_after,
                    duration,
                ))


        return next_item

    def shutdown(self):
        if not self._shutdown_event.is_set():
            self._shutdown_event.set()
            self._lock.release()

