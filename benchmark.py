import torch
import time
from combo_dataloader._combo_dataloader import ComboDataLoader, ComboDLTransform, DataLoaderType
import torchvision


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

    # # Start background process to write results to database
    # ctx = torch.multiprocessing.get_context('spawn')
    # results_queue = ctx.Queue()

    # print("enter a description of this benchmark")
    # experiment_description = input()

    # db_proc = Process(
    #     target=duckdb_wrapper.write_results,
    #     args=(
    #         results_queue,
    #         experiment_description,
    #     ),
    # )
    # db_proc.start()

    # Run trials
    for iteration in range(1):
        for n in range(11):
            pytorch = n
            dali = 10-n

            transform = ComboDLTransform(
                crop=112,
                mean=[0.43216, 0.394666, 0.37645],
                std=[0.22803 , 0.22145 , 0.216989],
                short_side_scale=128
            )
            #combo_dl = synthetic_data(96)

            combo_dl = ComboDataLoader(
                dataloaders=[DataLoaderType.PYTORCH, DataLoaderType.DALI],
                dataloader_portions=[pytorch, dali],
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

            clock_time, clips = run_trial(combo_dl)
            combo_dl.shutdown()
            # result = duckdb_wrapper.ComboFullBenchmarkEntry(
            #     iteration, pytorch, dali, clips, clock_time, queue_size
            # )
            # results_queue.put(result)
            trial = [iteration, pytorch, dali, clock_time, clips]
            print(trial)

    # results_queue.put("done")


def run_trial(dataloader):
    """
    Iterate once over the dataloader, applying the model on each batch.
    Return: the total clock time and the total number of clips processed.
    """

    # Load model
    model = torchvision.models.video.r3d_18()

    # Set to eval mode and move to GPU
    model = model.to("cuda")
    model = model.eval()

    perf_start = time.perf_counter()
    clips = 0

    for i, batch in enumerate(dataloader):
        # 8 batches, 3 channels, 16 frames, 256 size
        inputs = batch["frames"]

        clips += len(inputs)

        with torch.inference_mode():
            preds = model(inputs)

    clock_time = time.perf_counter() - perf_start
    return clock_time, clips

def synthetic_data(n):
    count = 0
    while count < n:
        yield {
            "frames":
            torch.cuda.FloatTensor(8, 3, 16, 112, 112).normal_()
        }
        count += 1

if __name__ == '__main__':
    main()
