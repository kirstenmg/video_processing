The `ComboDataloader` takes advantage of the CPU optimization of the existing
PyTorch video dataloader and the GPU optimization of NVIDIA's DALI video dataloader
to create a more efficient dataloader for video datasets. 

It runs a separate producer subprocess for each base dataloader, allowing better parallelism between the CPU and GPU. The pytorch dataloader also uses pushdown of video resize to the decoding step with the [decord](https://github.com/dmlc/decord) decoder, using [this fork](https://github.com/kirstenmg/pytorchvideo) of pytorchvideo to be able to pass arguments to the decoder used in the torch video dataloader.

## Getting started

Two sample notebooks with example usage are included:

`train_demo.ipynb`: demonstrates how to use the `ComboDataloader` in a video labeling train-test pipeline.

`speedup_demo.ipynb`: demonstrates the speedup of applying different optimizations to the `ComboDataloader`, and how to set up an optimal configuration.

To be able to use the resize pushdown optimization when using `decord` as the
PyTorch backend, use [this fork](https://github.com/kirstenmg/pytorchvideo) of
`pytorchvideo`. If using the provided Dockerfile, you may first need to uninstall the existing installation of `pytorchvideo` (`pip uninstall pytorchvideo`) before installing the forked module.

## Setting up Docker

To build the Docker image from the provided `Dockerfile`, run `docker build -t <image name> .`. Note that the Dockerfile includes dependencies needed for running the test notebooks that are not included in `requirements.txt`, since the `requirements.txt` includes `combo_dataloader` module dependencies only.


To use a GPU in the Docker container, make sure to pass `--runtime=nvidia`, `--ipc=host`,
and `--gpus '"capabilities=compute,utility,video"'` to `docker create`.
