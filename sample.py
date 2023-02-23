from combo_dataloader import ComboDataLoader, ComboDLTransform, DataLoaderType
import torchvision
import json
import torch
import time

def main():
  # Get video paths from annotation CSV
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

  transform = ComboDLTransform(
    crop=112,
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
    dali_pipeline_kwargs={"num_threads": 10}
  )

  # Load model and move to GPU
  model = torchvision.models.video.r3d_18()
  model = model.to("cuda")

  # Get classnames for kinetics dataset
  with open("kinetics_classnames.json", "r") as f:
      kinetics_classnames = json.load(f)

  # Create an id to label name mapping
  kinetics_id_to_classname = {}
  for k, v in kinetics_classnames.items():
      kinetics_id_to_classname[v] = str(k).replace('"', "")

  
  perf_start = time.perf_counter()
  for batch in dl:
    pass
    # preds = model(batch["frames"])
    # # Get the predicted classes
    # post_act = torch.nn.Softmax(dim=1)
    # preds = post_act(preds)
    # pred_classes = preds.topk(k=5).indices

    # # Map the predicted classes to the label names
    # pred_class_names = [kinetics_id_to_classname[int(i)] for i in pred_classes[0]]
    # print("Predicted labels: %s" % ", ".join(pred_class_names))
  
  print(time.perf_counter() - perf_start)


if __name__ == '__main__':
    main()
