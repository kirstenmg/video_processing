import torch
import datetime
from torch.profiler import profile, tensorboard_trace_handler
import time
import numpy as np
from dali_dataloader import DaliDataLoader
from combo_dataloader import ComboDataLoader

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

    with profile(
        record_shapes=False,
        on_trace_ready=tensorboard_trace_handler(f'{LOG_DIR}/{EXPERIMENT_DIR}')
    ) as prof:
        perf_start = time.perf_counter()
        proc_start = time.process_time()
        clips = 0
        print("beginning experiment")
        for i, batch in enumerate(dataloader):
            clips += batch_size
            inputs = batch["frames"]

            # Get the predicted classes
            with torch.inference_mode():
                preds = model(inputs)
                post_act = torch.nn.Softmax(dim=1)
                preds = post_act(preds)
                pred_classes = preds.topk(k=5).indices

            prof.step()
        clock_time = time.perf_counter() - perf_start
        process_time = time.process_time() - proc_start
        return clock_time, process_time, clips

if __name__ == '__main__':
    trials = []
    for iteration in range(1):
        for batch_size in [8]: #[3, 4, 5, 6, 7, 8]:
            for num_workers in [2]:#, 3, 5, 6, 7, 8, 9, 10]:
                clock_time, process_time, clips = run_experiment(
                    ComboDataLoader([DaliDataLoader(batch_size=batch_size, num_threads=num_workers)]),
                    iteration,
                    batch_size,
                    num_workers
                )
                trial = ["dali", iteration, batch_size, num_workers, clock_time, process_time, clips]
                print(trial)
                trials.append(trial)

    commas = [",".join([str(el) for el in row]) for row in trials]
    output = "\n".join(commas)
    print(output)

    now = str(datetime.datetime.now()).replace(" ", "_")
    with open(f'benchmark_results/{now}.csv', 'w') as file:
        file.write(output)



