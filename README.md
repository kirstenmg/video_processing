The `ComboDataloader` takes advantage of the CPU optimization of the existing
PyTorch video dataloader and the GPU optimization of NVIDIA's DALI video dataloader
to create a more efficient dataloader for video datasets. 

It runs a separate producer subprocess for each base dataloader, allowing better parallelism between the CPU and GPU. The pytorch dataloader also uses pushdown of video resize to the decoding step with the [decord](https://github.com/dmlc/decord) decoder, using [this fork](https://github.com/kirstenmg/pytorchvideo) of pytorchvideo to be able to pass arguments to the decoder used in the torch video dataloader.

## Getting started

First, install the `combo_dataloader` module: `python setup.py`

Several (quite similar) sample notebooks with example usage are included:

`speedup_demo.ipynb`: demonstrates the speedup of applying different optimizations to the `ComboDataloader`, and how to set up an optimal configuration.

`train_demo.ipynb`: demonstrates how to use the `ComboDataloader` in a video labeling train-test pipeline.

`train_speedup.ipynb`: demonstrates the speedup with the same configurations as in `speedup_demo.ipynb`, but in running a training pipeline rather than just dataloading.

`model_zoo.ipynb`: demonstrates using the `ComboDataloader` in pipelines with a few different models.

## Setting up Docker

To build the Docker image from the provided `Dockerfile`, run `docker build -t <image name> .`. Note that the Dockerfile includes dependencies needed for running the test notebooks that are not included in `requirements.txt`, since the `requirements.txt` includes `combo_dataloader` module dependencies only.


To use a GPU in the Docker container, make sure to pass `--runtime=nvidia`, `--ipc=host`,
and `--gpus '"capabilities=compute,utility,video"'` to `docker create`.

Creating the container:
```
docker create --name="container name"
	-v path/to/video_processing:/path/to/video_processing/in/container
	-v path/to/dataset:path/to/dataset/in/container:ro
	-i -t
	--runtime=nvidia --ipc=host --gpus'"capabilities=compute,utility,video"'
	<image name>
```

Starting the container:
```
docker start <container name>
```

Note that DALI does not support Python 3.10.

