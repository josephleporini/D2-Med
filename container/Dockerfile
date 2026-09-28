# D2 qualification container (Block T)
# Contract: ICD v1.0. Reads /app/input (ro), writes exactly /app/output/predictions.json.
# APL runs: docker run --rm --gpus all --network none -v /input:/app/input:ro -v /output:/app/output:rw IMAGE@DIGEST
#
# No NVIDIA base image is needed: the pinned torch cu130 wheels ship the CUDA 13.0 runtime
# (cuBLAS, cuDNN, NCCL) as pip packages, and the NVIDIA Container Toolkit injects the host
# driver at run time. Nothing here installs a kernel driver.

FROM python:3.11-slim-bookworm

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    HF_HUB_OFFLINE=1 \
    TRANSFORMERS_OFFLINE=1 \
    D2_INPUT=/app/input \
    D2_OUTPUT=/app/output \
    D2_MODEL_DIR=/app/model \
    D2_SCHEMA=/app/schema/qual-predictions.schema.json \
    D2_GPU_CAP_GB=14 \
    D2_BUDGET_S=3000

WORKDIR /app
COPY requirements.txt /app/requirements.txt
RUN pip install --index-url https://download.pytorch.org/whl/cu130 --extra-index-url https://pypi.org/simple \
        -r /app/requirements.txt \
    && find /usr/local/lib/python3.11 -name "*.pyc" -delete

COPY d2qual /app/d2qual
COPY schema /app/schema
COPY model /app/model

# Team identity is baked at build time (build-arg), not typed at run time.
ARG D2_TEAM_NAME=TEAM_NAME_TBD
ARG D2_TEAM_EMAIL=team@example.com
ARG D2_SUBMISSION_VERSION=0.1.0
ENV D2_TEAM_NAME=${D2_TEAM_NAME} D2_TEAM_EMAIL=${D2_TEAM_EMAIL} D2_SUBMISSION_VERSION=${D2_SUBMISSION_VERSION}

# Runs as root on purpose: the host-mounted /app/output must be writable regardless of host UID.
ENTRYPOINT ["python", "-m", "d2qual"]
