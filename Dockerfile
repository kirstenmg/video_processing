FROM pytorch/pytorch:1.13.0-cuda11.6-cudnn8-runtime
COPY requirements.txt requirements.txt
RUN apt-get update && apt-get install -y git
RUN python -m pip install -r requirements.txt
RUN pip install opencv-python-headless==4.7.0.72 jupyterlab==3.6.3 pytorch-lightning==1.9.5
