# -*- coding: utf-8 -*-
"""facebookresearch_pytorchvideo_slowfast.ipynb

# SlowFast

*Author: FAIR PyTorchVideo*

**SlowFast networks pretrained on the Kinetics 400 dataset**


### Example Usage

#### Imports

Load the model:
"""

import os
import random
import torch
from typing import Dict
import json
import urllib
from torchvision.transforms import Compose, Lambda, CenterCrop
from pytorchvideo.transforms import (
    ApplyTransformToKey,
    ShortSideScale,
    UniformTemporalSubsample,
    Normalize,
)
from combo_dataloader import ComboDataLoader, DataLoaderType, ComboDLTransform

if __name__ == "__main__":
    print("hi")

    # Choose the `slowfast_r50` model 
    model = torch.hub.load('facebookresearch/pytorchvideo', 'slowfast_r50', pretrained=True)

    """#### Setup

    Set the model to eval mode and move to desired device.
    """
    print("hi")

    # Set to GPU or CPU
    device = "cpu"
    model = model.eval()
    model = model.to(device)

    """Download the id to label mapping for the Kinetics 400 dataset on which the torch hub models were trained. This will be used to get the category label names from the predicted class ids."""

    with open("kinetics_classnames.json", "r") as f:
            kinetics_classnames = json.load(f)

    # Create an id to label name mapping
    kinetics_id_to_classname = {}
    for k, v in kinetics_classnames.items():
            kinetics_id_to_classname[v] = str(k).replace('"', "")

    """#### Define input transform"""

    side_size = 256
    mean = [0.45, 0.45, 0.45]
    mean = [item * 255 for item in mean]
    std = [0.225, 0.225, 0.225]
    std = [item * 255 for item in std]
    crop_size = 256
    num_frames = 32
    sampling_rate = 2
    frames_per_second = 30
    slowfast_alpha = 4

    print("hi2")
    class PackPathway(torch.nn.Module):
            """
            Transform for converting video frames as a list of tensors. 
            """
            def __init__(self):
                    super().__init__()
                    
            def forward(self, frames: torch.Tensor):
                    fast_pathway = frames
                    # Perform temporal sampling from the fast pathway.
                    slow_pathway = torch.index_select(
                            frames,
                            1,
                            torch.linspace(
                                    0, frames.shape[1] - 1, frames.shape[1] // slowfast_alpha
                            ).long(),
                    )
                    frame_list = [slow_pathway, fast_pathway]
                    return frame_list

    transform =  ApplyTransformToKey(
            key="video",
            transform=Compose(
                    [
                            UniformTemporalSubsample(num_frames),
                            Lambda(lambda x: x/255.0),
                            Normalize(mean, std),
                            ShortSideScale(
                                    size=side_size
                            ),
                            CenterCrop(crop_size),
                            PackPathway()
                    ]
            ),
    )

    # The duration of the input clip is also specific to the model.
    clip_duration = (num_frames * sampling_rate)/frames_per_second

    """#### Run Inference

    Download an example video.
    """

    with open("kinetics_classnames.json", "r") as f:
            kinetics_classnames_json = json.load(f)

    # Create a label name to id mapping
    kinetics_classnames_to_id = {}
    for k, v in kinetics_classnames_json.items():
            kinetics_classnames_to_id[str(k).replace('"', "")] = v

    # Create an id to label name mapping
    kinetics_id_to_classname = {}
    for k, v in kinetics_classnames_to_id.items():
            kinetics_id_to_classname[v] = k

    # These videos are corrupted and cannot be read
    # Ignore this for running on your own system
    null_videos = {
            "/home/maureen/kinetics/kinetics400_10classes/train/xxUezLcXkDs_000256_000266.mp4",
            "/home/maureen/kinetics/kinetics400_10classes/train/CUxsn4YXksI_000119_000129.mp4"
    }

    # Given an annotation file and a base path to the videos, load the video paths and labels
    def load_video_paths(annotation_file_path, video_base_path, shuffle=True):
        video_paths = []
        labels = []
        with open(annotation_file_path, 'r') as annotation_file:
            for i, line in enumerate(annotation_file):
                if i != 0: # skip column headers
                    line = annotation_file.readline()
                    if line:
                        label, youtube_id, time_start, time_end, split, is_cc = line.strip().split(',')
                        label_id = kinetics_classnames_to_id.get(label)
                        vpath = f'{video_base_path}/{split}/{youtube_id}_{int(time_start):06d}_{int(time_end):06d}.mp4'

                        if os.path.exists(vpath) and vpath not in null_videos:
                            video_paths.append(vpath)
                            labels.append(label_id)

        if shuffle:
            combined = list(zip(video_paths, labels))
            random.shuffle(combined)
            video_paths, labels = zip(*combined)

        return video_paths, labels

    train_paths, train_labels = load_video_paths(
            '/home/maureen/kinetics/kinetics400_10classes/annotations/train.csv',
            '/home/maureen/kinetics/kinetics400_10classes'
    )

    import slowfast_transform

    transform = ComboDLTransform(
            crop=256,
            mean=mean,
            std =std,
            short_side_scale=256
    )


    train_dl = ComboDataLoader(
            dataloaders=[DataLoaderType.PYTORCH],
            dataloader_portions=[1],
                  video_paths=train_paths,
labels=train_labels,
            transform=transform,
            stride=2,
            step=32,
            sequence_length=16,
            fps=30,
            batch_size=8,
            pytorch_dataloader_kwargs={"num_workers": 10},
            pytorch_dataset_kwargs=dict(decoder="decord", short_side_scale=128),
            pytorch_additional_transform=slowfast_transform.PackPathway()
    )

    inputs = next(iter(train_dl))["frames"]

    """#### Get Predictions"""

    # Pass the input clip through the model
    preds = model(inputs)

    # Get the predicted classes
    post_act = torch.nn.Softmax(dim=1)
    preds = post_act(preds)
    pred_classes = preds.topk(k=5).indices[0]

    # Map the predicted classes to the label names
    pred_class_names = [kinetics_id_to_classname[int(i)] for i in pred_classes]
    print("Top 5 predicted labels: %s" % ", ".join(pred_class_names))

    """### Model Description
    SlowFast model architectures are based on [1] with pretrained weights using the 8x8 setting
    on the Kinetics dataset. 

    | arch | depth | frame length x sample rate | top 1 | top 5 | Flops (G) | Params (M) |
    | --------------- | ----------- | ----------- | ----------- | ----------- | ----------- |  ----------- | ----------- |
    | SlowFast | R50   | 8x8                        | 76.94 | 92.69 | 65.71     | 34.57      |
    | SlowFast | R101  | 8x8                        | 77.90 | 93.27 | 127.20    | 62.83      |


    ### References
    [1] Christoph Feichtenhofer et al, "SlowFast Networks for Video Recognition"
    https://arxiv.org/pdf/1812.03982.pdf
    """

    print("hello")