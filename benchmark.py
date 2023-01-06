import torch
import datetime
from torch.profiler import profile, tensorboard_trace_handler
import time
from combo_dataloader import ComboDataLoader, DataLoaderType

LOG_DIR = "profiler_logs"
""" Train """
# Pass the input clip through the model
def run_experiment(dataloader, iteration, batch_size, num_threads):
    """ Set up pretrained model """
    model_name = "slow_r50"
    model = torch.hub.load("facebookresearch/pytorchvideo", model=model_name, pretrained=True)

    # Set to eval mode and move to desired device
    device = "cuda"
    model = model.to(device)
    model = model.eval()

    # Create dataloader from dataset
    now = str(datetime.datetime.now()).replace(" ", "_")
    EXPERIMENT_DIR = f'pytorch_slowr50_{now}_batch{batch_size}_threads{num_threads}_{iteration}'

    perf_start = time.perf_counter()
    proc_start = time.process_time()
    clips = 0
    print("beginning experiment")
    for i, batch in enumerate(dataloader):
        print(i)
        clips += batch_size
        inputs = batch["frames"]

        # Get the predicted classes
        time.sleep(0.1)
        with torch.inference_mode():
            preds = model(inputs)
            post_act = torch.nn.Softmax(dim=1)
            preds = post_act(preds)
            pred_classes = preds.topk(k=5).indices

    clock_time = time.perf_counter() - perf_start
    process_time = time.process_time() - proc_start
    return clock_time, process_time, clips

if __name__ == '__main__':
    """ Set up video paths """
    # Get video paths from annotation CSV
    ANNOTATION_FILE_PATH = "/home/maureen/kinetics/kinetics400/annotations/val.csv"
    VIDEO_BASE_PATH = "/home/maureen/kinetics/kinetics400"
    video_paths = []
    with open(ANNOTATION_FILE_PATH, 'r') as annotation_file:
        for i, line in enumerate(annotation_file):
            if i != 0: # skip column headers
                line = annotation_file.readline()
                label, youtube_id, time_start, time_end, split, is_cc = line.strip().split(',')
                vpath = f'{VIDEO_BASE_PATH}/{split}/{youtube_id}_{int(time_start):06d}_{int(time_end):06d}.mp4'
                video_paths.append(vpath)

    #video_paths = video_paths[:10]

    trial_setup = [
        ([DataLoaderType.PYTORCH, DataLoaderType.PYTORCH], "pytorch"),
        #([DataLoaderType.DALI, DataLoaderType.DALI], "dali"),
        #([DataLoaderType.PYTORCH, DataLoaderType.DALI], "dali_pytorch"),
    ]

    trials = []
    for iteration in range(1):
        for batch_size in [8]: #[3, 4, 5, 6, 7, 8]:
            for num_workers in [2]:#, 3, 5, 6, 7, 8, 9, 10]:
                for dl_list, str_desc in trial_setup:
                    combo_dl = ComboDataLoader(dl_list, video_paths)
                    clock_time, process_time, clips = run_experiment(
                        combo_dl,
                        iteration,
                        batch_size,
                        num_workers
                    )
                    trial = [str_desc, iteration, batch_size, num_workers, clock_time, process_time, clips]
                    print(trial)
                    trials.append(trial)

    commas = [",".join([str(el) for el in row]) for row in trials]
    output = "\n".join(commas)
    print(output)

    now = str(datetime.datetime.now()).replace(" ", "_")
    with open(f'benchmark_results/{now}.csv', 'w') as file:
        file.write(output)

