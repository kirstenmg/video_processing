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
from torch.profiler import profile, ProfilerActivity, tensorboard_trace_handler

""" Constants """
ANNOTATION_FILE_PATH = "/home/maureen/kinetics/kinetics400/annotations/val.csv"
VIDEO_BASE_PATH = "/home/maureen/kinetics/kinetics400"
LOG_DIR = "profiler_logs"
EXPERIMENT_DIR = "pytorch_slowr50_test"
BATCH_SIZE = 4
CROP_SIZE = 256
CLIP_DURATION = 10 # seconds
NUM_WORKERS = 4
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
num_frames = 8
sampling_rate = 8
frames_per_second = 30
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
    clip_sampler=pytorchvideo.data.make_clip_sampler("uniform", CLIP_DURATION),
    transform=transform,
    decode_audio=False
)

# Create dataloader from dataset
dataloader = torch.utils.data.DataLoader(
    dataset,
    batch_size=BATCH_SIZE,
    num_workers=NUM_WORKERS,
)

""" Set up pretrained model """
model_name = "slow_r50"
model = torch.hub.load("facebookresearch/pytorchvideo", model=model_name, pretrained=True)

# Set to eval mode and move to desired device
model = model.to(device)
model = model.eval()


""" Train """
# Pass the input clip through the model
dataloader = iter(dataloader)

with profile(
    record_shapes=False,
    on_trace_ready=tensorboard_trace_handler(f'{LOG_DIR}/{EXPERIMENT_DIR}')
) as prof:
    for batch in dataloader:
        inputs = batch["video"].to(device)
        preds = model(inputs)

        # Get the predicted classes
        post_act = torch.nn.Softmax(dim=1)
        preds = post_act(preds)
        pred_classes = preds.topk(k=5).indices

        # Map the predicted classes to the label names
        prof.step()
