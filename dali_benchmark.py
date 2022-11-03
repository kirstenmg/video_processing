from nvidia.dali import pipeline_def
import nvidia.dali.fn as fn
import nvidia.dali.types as types
from nvidia.dali.plugin.pytorch import DALIGenericIterator, LastBatchPolicy

"""Constants"""
BATCH_SIZE = 4
NUM_THREADS = 3

sequence_length=16
stride=2
step=(sequence_length * stride) // 2
device='gpu'
normalized=False

""" File paths """
ANNOTATION_FILE_PATH = "/home/maureen/kinetics/kinetics400/annotations/val.csv"
VIDEO_BASE_PATH = "/home/maureen/kinetics/kinetics400"
video_paths = []
with open(ANNOTATION_FILE_PATH, 'r') as annotation_file:
    for i, line in enumerate(annotation_file):
        if i != 0: # skip column headers
            line = annotation_file.readline()
            label, youtube_id, time_start, time_end, split, is_cc = line.strip().split(',')
            vpath = f'{VIDEO_BASE_PATH}/{split}/{youtube_id}_{int(time_start):06d}_{int(time_end):06d}.mp4'
            video_paths.append((vpath, {"name": vpath}))


""""Set up dataloader"""
def dali_transform(frames):
    frames = fn.crop_mirror_normalize(
        frames,
        dtype=types.FLOAT,
        output_layout="CFHW",
        crop=(112, 112),
        mean=[0.43216 * 255, 0.394666 * 255, 0.37645 * 255],
        std=[0.22803 * 255, 0.22145 * 255, 0.216989 * 255],
        mirror=False
    )
    return frames

resize_kwargs=dict(resize_shorter=128)

@pipeline_def
def create_pipeline():
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
        # filenames=video_paths,
        file_root="/home/maureen/kinetics400/val/",
        labels=[],
        name='reader',
    )
    frames = dali_transform(frames)

    return frames, label, timestamp

pipeline = create_pipeline(batch_size=BATCH_SIZE, num_threads=NUM_THREADS, device_id=0)
pipeline.build()

"""
"""


video_iterator = DALIGenericIterator(
            pipeline,
            ['frames', 'vid', 'frame_timestamp'],
            last_batch_policy=LastBatchPolicy.PARTIAL,
            reader_name='reader'
        )