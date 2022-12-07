"""
Dataloader that uses a producer/consumer model to parallelize multiple dataloaders.
"""

from typing import Dict, List
from threading import Thread, Event
from queue import Queue
from torch import Tensor
from dataloader import DataLoader


def populate_queue(queue: Queue, dl: DataLoader, event: Event) -> None:
    next_batch = dl.next_batch()
    while next_batch is not None and not event.is_set():
        queue.put(next_batch)
        next_batch = dl.next_batch()

class ComboDataLoader(DataLoader):
    def __init__(self, dataloaders: List[DataLoader]):
        self._batch_queue = Queue()  # stores Dict[str, Tensor]
        self._started = False
        self._shutdown_event = Event()
        self._dl_threads = [
            Thread(target=populate_queue, args=(self._batch_queue,dataloader, self._shutdown_event))
            for dataloader in dataloaders
        ]

    def next_batch(self) -> Dict[str, Tensor]:
        """
        Return the next batch of inputs. Tensor input is stored under the "video"
        key of the result dictionary.
        NOTE: may add labels to the output in the future.
        """

        if not self._started:
            for t in self._dl_threads:
                t.start()
            self._started = True

        # Return next item in queue, blocking
        return self._batch_queue.get();

    def shutdown(self):
        self._shutdown_event.set()


