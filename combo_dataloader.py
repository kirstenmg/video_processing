"""
Dataloader that uses a producer/consumer model to parallelize multiple dataloaders.
"""

from typing import Dict, List
from torch import Tensor, multiprocessing
from dataloader import DataLoader
from dali_dataloader import DaliDataLoader
import time


def populate_queue(queue: multiprocessing.Queue, shutdown_lock) -> None:
    dl = DaliDataLoader(8, 10)
    for batch in dl:
        queue.put(batch)

    queue.put("done")

    #TODO: don't terminate yet
    shutdown_lock.acquire()


class ComboDataLoader(DataLoader):
    def __init__(self, dataloaders: List[DataLoader]):
        ctx = multiprocessing.get_context('spawn')
        self._batch_queue = ctx.Queue()  # stores Dict[str, Tensor]
        self._shutdown_lock = ctx.Lock()
        self._shutdown_lock.acquire()
        self._started = False
        self._dl_processes = [
            ctx.Process(target=populate_queue, args=(self._batch_queue,self._shutdown_lock), daemon=True)
            for dataloader in dataloaders
        ]

    def __next__(self):
        """
        Return the next batch of inputs. Tensor input is stored under the "video"
        key of the result dictionary.
        NOTE: may add labels to the output in the future.
        """

        if not self._started:
            for t in self._dl_processes:
                t.start()
                print(f'pid {t.pid}')
            self._started = True

        # Return next item in queue, blocking
        next_item = self._batch_queue.get();
            
        if next_item == "done":
            raise StopIteration

        return next_item

