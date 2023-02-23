import pytorchvideo.data
import torch
from torchvision.transforms import Compose, CenterCrop
from pytorchvideo.transforms import (
    ApplyTransformToKey,
    ShortSideScale,
    UniformTemporalSubsample,
    Normalize
)
from ._dataloader import DataLoader, DataLoaderParams

""" Constants """
fps = 27 # due to variation in fps in dataset
clip_duration = 32 / fps # 30 fps, we want 32 frames per clip, so 32/30 seconds
stride = 32 / fps # seconds to offset next clip by; 24/30 to match dali setup

device="cuda"

# Set up transform
num_frames = 16
sampling_rate = 8

class PytorchDataloader(DataLoader):
    def __init__(
        self,
        params: DataLoaderParams
    ):
        if not params.pytorch_dataloader_kwargs:
            params.pytorch_dataloader_kwargs = dict()
        if not params.pytorch_dataset_kwargs:
            params.pytorch_dataset_kwargs = dict()

        # TODO: additional transform
        dataset_transform =  ApplyTransformToKey(
            key="video",
            transform=Compose(
                [
                    UniformTemporalSubsample(params.sequence_length),
                    Normalize(params.transform.mean, params.transform.std),
                    ShortSideScale(
                            size=params.transform.short_side_scale
                    ),
                    CenterCrop(params.transform.crop)
                ]
            ),
        )

        if params.labels:
            reformatted_video_paths = [(path, {"label": label}) for path, label in zip(params.video_paths, params.labels)]
        else:
            reformatted_video_paths = [(path, {}) for path in params.video_paths]

        dataset = pytorchvideo.data.LabeledVideoDataset(
            labeled_video_paths=reformatted_video_paths,
            # TODO: adjust correctly
            clip_sampler=pytorchvideo.data.make_clip_sampler(
                "uniform",
                params.stride * params.sequence_length / params.fps,
                params.step / params.fps,
            ),
            video_sampler=torch.utils.data.SequentialSampler,
            transform=dataset_transform,
            decode_audio=False,
            decoder="decord",
            **params.pytorch_dataset_kwargs
        )

        dataloader = torch.utils.data.DataLoader(
            dataset,
            batch_size=params.batch_size,
            **params.pytorch_dataloader_kwargs
        )

        self._iter = iter(dataloader)

    def __next__(self):
        batch = next(self._iter)
        inputs = batch["video"].to(device)

        if "labels" in batch:
            return {"frames": inputs, "labels": batch["labels"]}
        
        return {"frames": inputs}

