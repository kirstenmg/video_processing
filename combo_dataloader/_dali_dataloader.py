from nvidia.dali import pipeline_def
import nvidia.dali.fn as fn
import nvidia.dali.types as types
from nvidia.dali.plugin.pytorch import DALIGenericIterator, LastBatchPolicy
from ._dataloader import DataLoader, DataLoaderParams


"""
    Constants
"""
# Clip sampling
# 30 fps = 300 fpvideo
device='gpu'
normalized=False

@pipeline_def
def create_pipeline(params: DataLoaderParams):
    resize_args = dict()
    if params.transform.short_side_scale:
        resize_args["resize_shorter"] = params.transform.short_side_scale

    frames, label, timestamp = fn.readers.video_resize(
        **resize_args,
        device=device,
        sequence_length=params.sequence_length, # Frames to load per sequence
        stride=params.stride, # Distance between consecutive frames
        step=params.step, # Frame interval between each sequence
        normalized=normalized,
        random_shuffle=False,
        image_type=types.RGB,
        dtype=types.UINT8,
        initial_fill=None, # Only relevant when shuffle=True
        pad_last_batch=False,
        dont_use_mmap=True,
        skip_vfr_check=True,
        enable_timestamps=True,
        filenames=params.video_paths,
        labels=list(range(len(params.video_paths))),
        name='reader',
        interp_type=types.INTERP_NN,
        **params.dali_reader_kwargs,
    )

    crop_args = dict()
    if params.transform.crop:
        if type(params.transform.crop) == int:
          # Create a square from given crop value, since only a single value given
          crop_args["crop"] = (params.transform.crop, params.transform.crop)
        else:
          crop_args["crop"] = params.transform.crop

    frames = fn.crop_mirror_normalize(
        frames,
        dtype=types.FLOAT,
        output_layout="CFHW",
        **crop_args,
        mean=params.transform.mean,
        std=params.transform.std,
        mirror=False
    )
 
    if params.dali_additional_transform:
        frames = params.dali_additional_transform(frames)

    return frames, label, timestamp



class DaliDataLoader(DataLoader):
    def __init__(self, params: DataLoaderParams):

        if "num_threads" not in params.dali_pipeline_kwargs:
            # Default value of num_threads in pipeline_def may only be used with
            # serialized pipeline, so we need to set a value here
            params.dali_pipeline_kwargs["num_threads"] = 1

        pipeline = create_pipeline(params=params, batch_size=params.batch_size, device_id=0, **params.dali_pipeline_kwargs)
        pipeline.build()
        
        dali_iter = DALIGenericIterator(
            pipeline,
            ['frames', 'label', 'frame_timestamp'],
            last_batch_policy=LastBatchPolicy.PARTIAL,
            reader_name='reader'
        )

        self._iterator = iter(dali_iter)

    def __next__(self):
        batch = next(self._iterator)[0]
        batch["label"] = batch["label"].squeeze()
        return batch