import pytorchvideo.data
from typing import List
from dataloader import DataLoader
import torch
from torchvision.transforms import Compose, Lambda, CenterCrop
from pytorchvideo.transforms import (
    ApplyTransformToKey,
    ShortSideScale,
    UniformTemporalSubsample,
    Normalize
)

""" Constants """
frames_per_second = 30
LOG_DIR = "profiler_logs"
CROP_SIZE = 256
CLIP_DURATION = 32 / 30 # 30 fps, we want 32 frames per clip, so 32/30 seconds
CLIPS_PER_VIDEO = 10
device="cuda"

# Set up transform
side_size = 256
mean = [0.45, 0.45, 0.45]
std = [0.225, 0.225, 0.225]
crop_size = 256
num_frames = 16
sampling_rate = 8

class PytorchDataloader(DataLoader):
    def __init__(self, batch_size: int, num_threads: int, video_paths: List[str]):
        transform =  ApplyTransformToKey(
            key="video",
            transform=Compose(
                [
                    UniformTemporalSubsample(num_frames),
                    # Lambda(lambda x: x/255.0),
                    Normalize(mean, std),
                    ShortSideScale(
                            size=side_size
                    ),
                    CenterCrop(crop_size)
                ]
            ),
        )
        
        reformatted_video_paths = [(path, {}) for path in video_paths]

        dataset = pytorchvideo.data.LabeledVideoDataset(
            labeled_video_paths=reformatted_video_paths,
            clip_sampler=pytorchvideo.data.make_clip_sampler(
                "constant_clips_per_video",
                CLIP_DURATION,
                CLIPS_PER_VIDEO
                ),
            video_sampler=torch.utils.data.SequentialSampler,
            transform=transform,
            decode_audio=False
        )

        dataloader = torch.utils.data.DataLoader(
            dataset,
            batch_size=batch_size,
            num_workers=num_threads,
        )

        self._iter = iter(dataloader)

    def __next__(self):
        batch = next(self._iter)
        inputs = batch["video"].to(device)

        # TODO: include labels in the future; same for DALI dataloader
        return {"frames": inputs}

