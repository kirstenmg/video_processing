from nvidia.dali import pipeline_def
import nvidia.dali.fn as fn
import nvidia.dali.types as types
from nvidia.dali.plugin.pytorch import DALIGenericIterator, LastBatchPolicy
import torch
from torch.profiler import profile, tensorboard_trace_handler
import datetime
from collections import Counter
import time
import numpy as np

"""Constants"""

sequence_length=16
stride=2
step=27 # to match fps and give us 10 clips per video
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
            video_paths.append(vpath)

""""Set up dataloader"""
mean = [0.45, 0.45, 0.45]
std = [0.225, 0.225, 0.225]
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


resize_kwargs=dict(resize_shorter=256)
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
        filenames=video_paths,
        labels=list(range(len(video_paths))),
        name='reader',
    )
    frames = dali_transform(frames)

    return frames, label, timestamp
""" Set up pretrained model """
model_name = "slow_r50"
model = torch.hub.load("facebookresearch/pytorchvideo", model=model_name, pretrained=True)

# Set to eval mode and move to desired device
model = model.to("cuda")
model = model.eval()


""" Train """
# Pass the input clip through the model

def run_experiment(iteration, batch_size, num_threads):
    LOG_DIR = "profiler_logs"
    EXPERIMENT_DIR = f'dali_slowr50_{str(datetime.datetime.now())}_batch{batch_size}_threads{num_threads}_{iteration}'

    pipeline = create_pipeline(batch_size=batch_size, num_threads=num_threads, device_id=0)
    pipeline.build()

    video_iterator = DALIGenericIterator(
        pipeline,
        ['frames', 'vid', 'frame_timestamp'],
        last_batch_policy=LastBatchPolicy.PARTIAL,
        reader_name='reader'
    )

    clock_time = -1
    process_time = -1
    clips = 0
    with profile(
        record_shapes=False,
        on_trace_ready=tensorboard_trace_handler(f'{LOG_DIR}/{EXPERIMENT_DIR}')
    ) as prof:
        # TODO: what to include in timing? I'm doing this section because it is
        # what gets profiled.
        perf_start = time.perf_counter()
        proc_start = time.process_time()
        for batch in video_iterator:
            clips += batch_size
            inputs = batch[0]["frames"]
            preds = model(inputs)

            # Get the predicted classes
            with torch.inference_mode():
                post_act = torch.nn.Softmax(dim=1)
                preds = post_act(preds)
                pred_classes = preds.topk(k=5).indices
            prof.step()
        clock_time = time.perf_counter() - perf_start
        process_time = time.process_time() - proc_start
    return clock_time, process_time, clips
    

trials = []
for iteration in range(1):
    for batch_size in [8]: #, 2, 4, 8]:
        for num_workers in [4]: #, 4]:
            clock_time, process_time, clips = run_experiment(iteration, batch_size, num_workers)
            trial = [iteration, batch_size, num_workers, clock_time, process_time, clips]
            print(trial)
            trials.append(trial)

# np_trials = np.array(trials)
filename = f'benchmark/{str(datetime.datetime.now())}.csv'.replace(" ", "_")
lines = [
    ",".join(str(trial)) for trial in trials
    ]
output = "\n".join(lines)

with open(filename, 'w') as f:
    f.write(output)
