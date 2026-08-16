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

FROM pytorch/pytorch:2.5.1-cuda12.4-cudnn9-runtime

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
