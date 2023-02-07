import duckdb
from multiprocessing import Queue
from dataclasses import dataclass
from typing import Any, List, ClassVar
import time
from datetime import datetime

quote = lambda x: f'\'{x}\''

def write_results(queue: Queue, description: str):
  con = duckdb.connect(database="benchmark.duckdb", read_only=False) 
  # This experiment's id (timestamp)
  eid = int(round(time.time() * 1000))

  # Add entry for this experiment
  write_result(
    con,
    eid,
    ExperimentEntry(description)
  )

  result = queue.get()
  while result != "done":
    write_result(con, eid, result)
    result = queue.get()


def write_result(con, eid: int, result):
  values = ", ".join([str(eid), *[str(val) for val in result.get_values()]])
  con.execute(
    f'INSERT INTO {result.table_name} VALUES ({values})'
  )

"""
Schema
Correspond to the schema of tables in the database
Excludes eid, this is automatically added to each
"""

@dataclass
class ExperimentEntry:
  table_name: ClassVar[str] = "experiment"
  description: str

  def get_values(self) -> List[Any]:
    return [f'\'{self.description}\'']


@dataclass
class ComboFullBenchmarkEntry:
  table_name: ClassVar[str] = "combo_full_benchmark"

  iteration: int
  pytorch_portion: int
  dali_portion: int
  clip_count: int
  clock_time: float
  max_q_size: int

  def get_values(self) -> List[Any]:
    return [
      self.iteration,
      self.pytorch_portion,
      self.dali_portion,
      self.clip_count,
      self.clock_time,
      self.max_q_size,
    ]

@dataclass
class QueueBlockEntry:
  table_name: ClassVar[str] = "queue_block"

  operation: str
  pid: int
  dataloader: str
  start_time: datetime
  size_before: int
  size_after: int
  duration: float

  def get_values(self) -> List[Any]:
    return [
      quote(self.operation),
      self.pid,
      quote(self.dataloader),
      quote(self.start_time.isoformat()),
      self.size_before,
      self.size_after,
      self.duration,
    ]