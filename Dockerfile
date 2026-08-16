# Reproducible inference environment.
#
#   docker build -t semicon-restore .
#   docker run --gpus all --rm \
#       -v /path/to/test_images:/data/in \
#       -v /path/to/results:/data/out \
#       semicon-restore
#
# Without a GPU, drop `--gpus all` -- evaluate.py falls back to CPU
# automatically (slower, identical output).
#
# Extra flags pass straight through, e.g. for maximum throughput:
#   docker run --gpus all --rm -v ...:/data/in -v ...:/data/out \
#       semicon-restore --input_dir /data/in --output_dir /data/out --tta_views 1
#
# The weights are committed to this repository as four checksum-verified
# parts, so the image is self-contained: nothing is downloaded at run time.

# CUDA 12.8 is required, not incidental. PyTorch wheels built against CUDA 12.4
# and earlier contain no kernels for compute capability 12.0 (Blackwell, e.g.
# RTX 50-series), and fail at the first conv with:
#     CUDA error: no kernel image is available for execution on the device
# The cu128 builds cover sm_70 through sm_120, so this image runs on both older
# datacentre cards (V100/A100/H100) and current consumer ones.
FROM pytorch/pytorch:2.8.0-cuda12.8-cudnn9-runtime

WORKDIR /app

# Dependencies first so this layer caches across source edits. The base image
# already satisfies torch/torchvision; pip verifies rather than reinstalls.
# pytest is not in requirements.txt (it is not needed to run the model) but is
# needed for the build-time check below.
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt pytest

COPY . .

# Fail the build rather than the reviewer's run if anything is broken. This
# also proves the weight parts survived the COPY intact, since the tests
# checksum them.
#
# The build host has no GPU, so this necessarily runs CPU-only; the suite is
# device-agnostic and passes either way (verified with CUDA_VISIBLE_DEVICES="").
RUN python -m pytest tests -q

ENV PYTHONUNBUFFERED=1

ENTRYPOINT ["python", "evaluate.py"]
CMD ["--input_dir", "/data/in", "--output_dir", "/data/out"]
