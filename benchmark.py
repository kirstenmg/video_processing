import torch
import time
from combo_dataloader import ComboDataLoader, DataLoaderType
from multiprocessing import Process, Queue
import duckdb_wrapper
import torchvision


DB_BENCHMARK = False

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

    video_paths += video_paths

    # Start background process to write results to database
    ctx = torch.multiprocessing.get_context('spawn')
    results_queue = ctx.Queue()

    print("enter a description of this benchmark")
    experiment_description = input()

    db_proc = Process(
        target=duckdb_wrapper.write_results,
        args=(
            results_queue,
            experiment_description,
        ),
    )
    db_proc.start()

    # Run trials
    for iteration in range(3):
        for n in [0, 10, 20, 30, 32, 34, 36, 38, 40, 50, 60, 70, 80, 90, 100]:
            queue_size = 50
            pytorch = n
            dali = 100-n

            combo_dl = ComboDataLoader(
                [DataLoaderType.PYTORCH, DataLoaderType.DALI],
                video_paths,
                [pytorch, dali],
                queue_size,
                results_queue,
            )
            clock_time, clips = run_trial(combo_dl)
            combo_dl.shutdown()
            result = duckdb_wrapper.ComboFullBenchmarkEntry(
                iteration, pytorch, dali, clips, clock_time, queue_size
            )
            results_queue.put(result)
            trial = [iteration, pytorch, dali, clock_time, clips]
            print(trial)

    results_queue.put("done")


def run_trial(dataloader):
    """
    Iterate once over the dataloader, applying the model on each batch.
    Return: the total clock time and the total number of clips processed.
    """

    # Load model
    model_name = "slow_r50"
    model = torch.hub.load("facebookresearch/pytorchvideo", model=model_name, pretrained=True)

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
            torch.cuda.FloatTensor(8, 3, 16, 256, 256).normal_()
        }
        count += 1

if __name__ == '__main__':
    main()
