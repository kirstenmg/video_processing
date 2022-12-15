import pytorchvideo.data
import torch
from torchvision.transforms import Compose, Lambda
from pytorchvideo.transforms import (
    ApplyTransformToKey,
    ShortSideScale,
    UniformTemporalSubsample
)
from torchvision.transforms._transforms_video import (
    CenterCropVideo,
    NormalizeVideo,
)
import time
import numpy as np

""" Constants """
frames_per_second = 30
ANNOTATION_FILE_PATH = "/home/maureen/kinetics/kinetics400/annotations/val.csv"
VIDEO_BASE_PATH = "/home/maureen/kinetics/kinetics400"
LOG_DIR = "profiler_logs"
CROP_SIZE = 256
CLIP_DURATION = 32 / 30 # 30 fps, we want 32 frames per clip, so 32/30 seconds
CLIPS_PER_VIDEO = 10
device="cuda"

""" Set up dataloader """

# Get video paths from annotation CSV
video_paths = []
with open(ANNOTATION_FILE_PATH, 'r') as annotation_file:
    for i, line in enumerate(annotation_file):
        if i != 0: # skip column headers
            line = annotation_file.readline()
            label, youtube_id, time_start, time_end, split, is_cc = line.strip().split(',')
            vpath = f'{VIDEO_BASE_PATH}/{split}/{youtube_id}_{int(time_start):06d}_{int(time_end):06d}.mp4'
            video_paths.append((vpath, {"name": vpath}))

# Set up transform
side_size = 256
mean = [0.45, 0.45, 0.45]
std = [0.225, 0.225, 0.225]
crop_size = 256
num_frames = 16
sampling_rate = 8
transform =  ApplyTransformToKey(
    key="video",
    transform=Compose(
        [
            UniformTemporalSubsample(num_frames),
            Lambda(lambda x: x/255.0),
            NormalizeVideo(mean, std),
            ShortSideScale(
                size=side_size
            ),
            CenterCropVideo(crop_size=(crop_size, crop_size))
        ]
    ),
)

# Create dataset
dataset = pytorchvideo.data.LabeledVideoDataset(
    labeled_video_paths=video_paths,
    clip_sampler=pytorchvideo.data.make_clip_sampler(
        "constant_clips_per_video",
        CLIP_DURATION,
        CLIPS_PER_VIDEO
        ),
    transform=transform,
    decode_audio=False
)

""" Train """
# Pass the input clip through the model
def get_dataloader(batch_size, num_threads):
    # Create dataloader from dataset
    return torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=num_threads,
    )

get_input = lambda batch: batch["video"].to("cuda")
