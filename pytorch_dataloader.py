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
fps = 27 # due to variation in fps in dataset
clip_duration = 32 / fps # 30 fps, we want 32 frames per clip, so 32/30 seconds
stride = 32 / fps # seconds to offset next clip by; 24/30 to match dali setup

device="cuda"

# Set up transform
side_size = 128
mean=[0.43216, 0.394666, 0.37645]
std=[0.22803 , 0.22145 , 0.216989 ]
crop_size = 112
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
                "uniform",
                clip_duration,
                stride,
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

