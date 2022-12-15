from nvidia.dali import pipeline_def
import nvidia.dali.fn as fn
import nvidia.dali.types as types
from nvidia.dali.plugin.pytorch import DALIGenericIterator, LastBatchPolicy
from dataloader import DataLoader
from typing import List, Dict


"""
    Constants
"""
# Clip sampling
sequence_length=16
stride=2
step=27 # to match fps and give us 10 clips per video
device='gpu'
normalized=False

# Transform
mean = [0.45, 0.45, 0.45]
std = [0.225, 0.225, 0.225]
resize_kwargs=dict(resize_shorter=256)

def dali_transform(frames):
    frames = fn.crop_mirror_normalize(
        frames,
        dtype=types.FLOAT,
        output_layout="CFHW",
        crop=(256, 256),
        mean=[m * 255 for m in mean],
        std=[s * 255 for s in std],
        mirror=False
    )
    return frames

@pipeline_def
def create_pipeline(video_paths: List[str]):
    # TODO: use passed-in paths rather than hardcoding within this file
    frames, label, timestamp = fn.readers.video_resize(
        **resize_kwargs,
        device=device,
        sequence_length=sequence_length, # Frames to load per sequence
        stride=stride, # Distance between consecutive frames
        step=step, # Frame interval between each sequence
        normalized=normalized,
        random_shuffle=False,
        image_type=types.RGB,
        dtype=types.UINT8,
        initial_fill=None, # Only relevant when shuffle=True
        pad_last_batch=False,
        dont_use_mmap=True,
        skip_vfr_check=True,
        enable_timestamps=True,
        filenames=video_paths,
        labels=list(range(len(video_paths))),
        name='reader',
    )
    frames = dali_transform(frames)

    return frames, label, timestamp



class DaliDataLoader(DataLoader):
    def __init__(self, batch_size: int, num_threads: int, video_paths: List[str]):
        # TODO: consider adding more parameters rather than hardcoding a bunch
        # of constants
        pipeline = create_pipeline(video_paths=video_paths, batch_size=batch_size, num_threads=num_threads, device_id=0)
        pipeline.build()
        
        dali_iter = DALIGenericIterator(
            pipeline,
            ['frames', 'vid', 'frame_timestamp'],
            last_batch_policy=LastBatchPolicy.PARTIAL,
            reader_name='reader'
        )

        self._iterator = iter(dali_iter)

    def __next__(self):
        batch = next(self._iterator)
        return batch[0]