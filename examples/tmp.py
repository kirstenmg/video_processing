from combo_dataloader import ComboDataLoader, ComboDLTransform, DataLoaderType
import torchvision
import time
import torch.nn as nn
import pytorch_lightning
import torch
import json
from typing import List, Tuple
import os
import torchmetrics
import random
import slowfast_transform
dali_portion = 30
pytorch_portion = 70

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

null_videos = {
    "/home/maureen/kinetics/kinetics400_10classes/train/xxUezLcXkDs_000256_000266.mp4",
    "/home/maureen/kinetics/kinetics400_10classes/train/CUxsn4YXksI_000119_000129.mp4"
}
# Given an annotation file and a base path to the videos, load the video paths and labels
def load_video_paths(annotation_file_path, video_base_path, shuffle=True) -> Tuple[List[str], List[int]]:
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

train_paths = train_paths[:100]
train_labels = train_labels[:100]
transform = ComboDLTransform(
    crop=256,
    mean=[0.45, 0.45, 0.45],
    std = [0.225, 0.225, 0.225],
    short_side_scale=256
)


if __name__ == "__main__":
  train_dl = ComboDataLoader(
      dataloaders=[DataLoaderType.DALI],#, DataLoaderType.DALI],
      dataloader_portions=[pytorch_portion],#, dali_portion],
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
      dali_pipeline_kwargs={
          "num_threads": 10,
          "exec_async": False,
          "exec_pipelined": False,
      },
      dali_additional_transform=slowfast_transform.dali_pack_pathway,
      pytorch_additional_transform=slowfast_transform.PackPathway(),
  )

  for batch in train_dl:
    print(batch["frames"].shape)
    pass
