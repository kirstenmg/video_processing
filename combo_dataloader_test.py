import pytorchvideo.data
import numpy as np
import torch
import torchvision
from torchvision.transforms import Compose, CenterCrop
from pytorchvideo.transforms import (
    ApplyTransformToKey,
    ShortSideScale,
    UniformTemporalSubsample,
    Normalize
)
import nvidia.dali.fn as fn
import nvidia.dali.types as types
from typing import Dict, List, Any, Callable
from combo_dataloader import ComboDataLoader, ComboDLTransform, DataLoaderType

"""
Test functions
"""
# Tests that the inputs produced by a default pytorch dataloader
# are the same as the inputs produced by a ComboDataLoader with a single
# pytorch process, given the same settings
def test_default_pytorch_vs_combo_inputs(video_paths, model):
  o1 = get_inputs(model, get_default_pytorch_dataloader(decord=False, vids=video_paths), batch_to_input=lambda x: x["video"].to("cuda"))
  o2 = get_inputs(model, get_combo_pytorch_dataloader(decord=False, vids=video_paths), batch_to_input=lambda x: x["frames"])
  compare_outputs(o1, o2)


# Tests that constructing a combo dataloader and getting a batch succeeds when
# omitting optional arguments
def test_constructor_optional_args(video_paths):
  dl = ComboDataLoader(
     dataloaders=[DataLoaderType.DALI, DataLoaderType.PYTORCH],
     dataloader_portions=[1, 1],
     video_paths=video_paths,
     sequence_length=16,
     fps=32,
  )
  next(iter(dl))


def test_pytorch_additional_transform(video_paths):
  transform = ComboDLTransform(
      mean=[0.43216, 0.394666, 0.37645],
      std=[0.22803 , 0.22145 , 0.216989],
      short_side_scale=128
  )

  dl = ComboDataLoader(
      dataloaders=[DataLoaderType.PYTORCH],
      dataloader_portions=[1],
      video_paths=video_paths,
      transform=transform,
      stride=2,
      step=32,
      sequence_length=16,
      fps=32,
      batch_size=8,
      pytorch_dataloader_kwargs={"num_workers": 10},
      pytorch_additional_transform=CenterCrop(112),
  )

  # Check that input was cropped
  input = next(iter(dl))
  assert(list(input["frames"][0][0][0].shape) == [112, 112])


def additional_transform(frames):
  return fn.crop(
      frames,
      dtype=types.FLOAT,
      crop=(112, 112)
  )

def test_dali_additional_transform(video_paths):
  transform = ComboDLTransform(
      mean=[0.43216, 0.394666, 0.37645],
      std=[0.22803 , 0.22145 , 0.216989],
      short_side_scale=128
  )

  dl = ComboDataLoader(
      dataloaders=[DataLoaderType.DALI],
      dataloader_portions=[1],
      video_paths=video_paths,
      transform=transform,
      stride=2,
      step=32,
      sequence_length=16,
      fps=32,
      batch_size=8,
      dali_additional_transform=additional_transform, 
  )

  # Check that input was cropped
  input = next(iter(dl))
  assert(list(input["frames"][0][0][0].shape) == [112, 112])


"""
Helper functions to get and compare inputs/features
"""

# Get the features or raw inputs and aggregate by video path using agg function
# features: True if getting features by applying model, false if getting just inputs
# produced by model
def _get_features_or_inputs(model, dataloader, batch_to_input: Callable, get_features, agg: Callable = np.mean):
  video_to_all_data: Dict[str, List[Any]] = dict()
  for batch in dataloader:
    data = batch_to_input(batch)
    paths = batch["video_path"]

    if get_features:
      data = model(input)
      # post_act = torch.nn.Softmax(dim=1)
      # features = post_act(features)
    
    for path, data_list in zip(paths, data.tolist()):
      if path in video_to_all_data:
        video_to_all_data[path].append(data_list)
      else:
        video_to_all_data[path] = [data_list]

  return {
    video: agg(data, axis=0)
    for video, data in video_to_all_data.items()
  }


# Get the model inputs produced by dataloader and aggregate (max) by video path
# Aggregation doesn't have any logical meaning, but is done because the batches
# may be produced in different orders on different runs so aggregation provides
# a more consistent result
def get_inputs(model, dataloader, batch_to_input):
  return _get_features_or_inputs(model, dataloader, batch_to_input, get_features=False, agg=np.max)


def get_features(model, dataloader, batch_to_input):
  return _get_features_or_inputs(model, dataloader, batch_to_input, get_features=True, agg=np.mean)


# For each video, compare the aggregated outputs using allclose
def compare_outputs(video_to_outputs1, video_to_outputs2):
  diffs = []
  for v1, f1 in video_to_outputs1.items():
    f2 = video_to_outputs2[v1]
    diff = f1 - f2 
    diffs.append(max(diff.flatten()))
    if not np.allclose(f1, f2, atol=0.05):
      print("Error: Diverges for video " + v1)
  return diffs


# Compare the output of the two dataloaders, aggregated per video since order
# of dataloader outputs may be different
# Error if the output is not close
def test_output(dl1, batch_to_input1, dl2, batch_to_input2):
  inputs1 = dict()
  for batch in dl1:
      inputs1.append((batch["video_path"], np.array(batch_to_input1(batch).tolist())))
  inputs2 = dict()
  for batch in dl2:
      inputs2.append((batch["video_path"], np.array(batch_to_input2(batch).tolist())))


  for i1, i2 in zip(inputs1, inputs2):
    if not np.allclose(i1, i2):
      print("Input diverges")

"""
Helper functions to get dataloaders in certain configurations
"""

def get_pytorch_dataloader(
  *,
  sequence_length,
  mean,
  std,
  crop,
  short_side_scale,
  video_paths,
  use_decord,
  batch_size,
  num_workers,
):
  transforms = [
      ShortSideScale(short_side_scale),
      UniformTemporalSubsample(sequence_length),
      Normalize(mean, std),
      CenterCrop(crop),
  ]

  dataset_transform =  ApplyTransformToKey(
      key="video",
      transform=Compose(transforms),
  )

  decoder_args = dict()
  if use_decord:
    decoder_args = dict(
      decoder="decord",
      short_side_scale=short_side_scale
    )
  
  stride = 2
  step = 32
  dataset = pytorchvideo.data.LabeledVideoDataset(
      labeled_video_paths=video_paths,
      clip_sampler=pytorchvideo.data.make_clip_sampler(
          "uniform",
          stride * sequence_length / 32,
          step / 32,
      ),
      video_sampler=torch.utils.data.SequentialSampler,
      transform=dataset_transform,
      decode_audio=False,
      **decoder_args,
  )

  return torch.utils.data.DataLoader(
      dataset,
      batch_size=batch_size,
      num_workers=num_workers,
  )


def get_default_pytorch_dataloader(decord, vids): 
  return get_pytorch_dataloader(
    sequence_length=16,
    crop=112,
    mean=[0.43216, 0.394666, 0.37645],
    std=[0.22803 , 0.22145 , 0.216989],
    short_side_scale=128,
    video_paths=[(video, {"video_path": video}) for video in vids],
    use_decord=decord,
    batch_size=8,
    num_workers=10,
  )


def get_combo_pytorch_dataloader(decord, vids):
  transform = ComboDLTransform(
      crop=112,
      mean=[0.43216, 0.394666, 0.37645],
      std=[0.22803 , 0.22145 , 0.216989],
      short_side_scale=128
  )
  pd_kwargs = dict()
  if decord:
      pd_kwargs = dict(decoder="decord", short_side_scale=128)
  return ComboDataLoader(
      dataloaders=[DataLoaderType.PYTORCH],
      dataloader_portions=[1],
      video_paths=vids,
      transform=transform,
      stride=2,
      step=32,
      sequence_length=16,
      fps=32,
      batch_size=8,
      pytorch_dataset_kwargs=pd_kwargs,
      pytorch_dataloader_kwargs={"num_workers": 10},
  )

"""
Main
"""

def main():
  # Use Kinetics dataset
  # Pool features for each video by taking an element-wise mean
  # Test using all-close


  # Get video paths
  annotation_file_path = "/home/maureen/kinetics/kinetics400/annotations/val.csv"
  video_base_path = "/home/maureen/kinetics/kinetics400"
  video_paths = []
  with open(annotation_file_path, 'r') as annotation_file:
      for i, line in enumerate(annotation_file):
          if i != 0: # skip column headers
              line = annotation_file.readline()
              label, youtube_id, time_start, time_end, split, is_cc = line.strip().split(',')
              vpath = f'{video_base_path}/{split}/{youtube_id}_{int(time_start):06d}_{int(time_end):06d}.mp4'
              video_paths.append(vpath)

  # Load model and move to GPU
  model = torchvision.models.video.r3d_18()
  model = model.to("cuda")

  test_dali_additional_transform(video_paths[:10])
  test_default_pytorch_vs_combo_inputs(video_paths[:10], model)
  test_constructor_optional_args(video_paths[:10])
  test_pytorch_additional_transform(video_paths[:10])
  print("tests passed")


if __name__ == '__main__':
    main()

