FROM pytorch/pytorch
COPY requirements.txt requirements.txt
RUN python -m pip install -r requirements.txt
RUN pip install --extra-index-url https://developer.download.nvidia.com/compute/redist --upgrade nvidia-dali-cuda110
