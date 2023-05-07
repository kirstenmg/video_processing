import pytorchvideo.data
import torch
from torchvision.transforms import Compose, CenterCrop
from pytorchvideo.transforms import (
    ApplyTransformToKey,
    ShortSideScale,
    UniformTemporalSubsample,
    Normalize,
)
from ._dataloader import DataLoader, DataLoaderParams


device="cuda"

class PytorchDataLoader(DataLoader):
    def __init__(
        self,
        params: DataLoaderParams
    ):
        if not params.pytorch_dataloader_kwargs:
            params.pytorch_dataloader_kwargs = dict()
        if not params.pytorch_dataset_kwargs:
            params.pytorch_dataset_kwargs = dict()

        transforms = []

        if params.transform.short_side_scale and (params.pytorch_dataset_kwargs.get("decoder") != "decord" or \
                params.pytorch_dataset_kwargs.get("width", -1) < 0 or \
                params.pytorch_dataset_kwargs.get("height", -1) < 0):
            # Resize was not pushed down
            transforms.append(
                ShortSideScale(
                    size=params.transform.short_side_scale
                )
            )

        transforms.extend([
            UniformTemporalSubsample(params.sequence_length),
            Normalize(params.transform.mean, params.transform.std),
        ])

        if params.transform.crop:
            transforms.append(CenterCrop(params.transform.crop))

        if params.pytorch_additional_transform:
            transforms.append(params.pytorch_additional_transform)

        dataset_transform =  ApplyTransformToKey(
            key="video",
            transform=Compose(transforms),
        )

        if params.labels:
            reformatted_video_paths = [(path, {"label": label, "video_path": path}) for path, label in zip(params.video_paths, params.labels)]
        else:
            reformatted_video_paths = [(path, {"video_path": path}) for path in params.video_paths]

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
            **params.pytorch_dataset_kwargs
        )

        dataloader = torch.utils.data.DataLoader(
            dataset,
            batch_size=params.batch_size,
            **params.pytorch_dataloader_kwargs
        )

        self._dataloader = dataloader

    def __iter__(self):
        return _PytorchIter(iter(self._dataloader))


class _PytorchIter():
    def __init__(self, dataloader_iter):
        self._iter = dataloader_iter

    def __iter__(self):
        return self

    def __next__(self):
        batch = next(self._iter)
        inputs = batch["video"].to(device)

        batch["frames"] = inputs
        del batch["video"]
        
        return batch



