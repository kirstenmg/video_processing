FROM pytorch/pytorch:1.13.0-cuda11.6-cudnn8-runtime
COPY requirements.txt requirements.txt
RUN python --version
RUN apt-get update && apt-get install -y git
RUN python -m pip install -r requirements.txt
RUN python --version
RUN pip install opencv-python-headless==4.7.0.72
RUN pip install jupyterlab==3.6.3
RUN pip install pytorch-lightning==1.9.5
RUN python --version
