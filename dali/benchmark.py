from nvidia.dali import pipeline_def
import nvidia.dali.fn as fn
import nvidia.dali.types as types

def dali_transform(frames):
    frames = fn.crop_mirror_normalize(
        frames,
        dtype=types.FLOAT,
        output_layout="CFHW",
        crop=(112, 112),
        mean=[0.43216 * 255, 0.394666 * 255, 0.37645 * 255],
        std=[0.22803 * 255, 0.22145 * 255, 0.216989 * 255],
        mirror=False
    )
    return frames

@pipeline_def
def create_pipeline():
    frames, label, timestamp = fn.readers.video_resize(
        device=device,
        **resize_kwargs,
        sequence_length=sequence_length,
        stride=stride,
        step=step,
        normalized=normalized,
        random_shuffle=False,
        image_type=types.RGB,
        dtype=types.UINT8,
        initial_fill=None, # Only relevant when shuffle=True
        pad_last_batch=False,
        dont_use_mmap=True,
        skip_vfr_check=True,
        enable_timestamps=True,
        filenames=files,
        labels=vids,
        name='reader',
    )
    frames = dali_transform(frames)

    return frames, label, timestamp

    # Add args batch_size, num_threads, device_id for @pipeline_def
    logging.debug(f'Creating dali pipeline with batch_size={batch_size}, num_threads={num_threads}')
    pipeline = create_pipeline(batch_size=batch_size, num_threads=num_threads, device_id=0)
    pipeline.build()
    return pipeline
resize_kwargs=dict(
                resize_shorter=128
            )

sequence_length=16,
            stride=2,
            step=(sequence_length * stride) // 2


device='gpu'
normalized=False

video_iterator = dali_pytorch.DALIGenericIterator(
            built_pipeline,
            ['frames', 'vid', 'frame_timestamp'],
            last_batch_policy=dali_pytorch.LastBatchPolicy.PARTIAL,
            reader_name='reader'
        )