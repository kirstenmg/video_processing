FROM pytorch/pytorch
COPY requirements.txt requirements.txt
RUN python -m pip install -r requirements.txt
RUN pip install opencv-python-headless
RUN pip install jupyterlab
