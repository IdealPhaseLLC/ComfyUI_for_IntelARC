FROM ubuntu:24.04

ARG DEBIAN_FRONTEND=noninteractive
ARG COMFYUI_REF=master

# Intel OMIX supplies the Level Zero/OpenCL user-space runtime for Arc Pro B50.
# The Linux kernel driver itself remains on the Docker host.
RUN apt-get update \
 && apt-get install -y --no-install-recommends \
        ca-certificates \
        git \
        gnupg \
        wget \
        python3 \
        python3-venv \
        ffmpeg \
        libgl1 \
        libglib2.0-0 \
        libgomp1 \
 && wget -qO - https://repositories.intel.com/gpu/intel-graphics.key \
      | gpg --yes --dearmor --output /usr/share/keyrings/intel-graphics.gpg \
 && echo 'deb [arch=amd64 signed-by=/usr/share/keyrings/intel-graphics.gpg] https://repositories.intel.com/gpu/ubuntu noble/intel-omix/0.4 unified' \
      > /etc/apt/sources.list.d/intel-omix.list \
 && apt-get update \
 && apt-get install -y --no-install-recommends intel-omix \
 && rm -rf /var/lib/apt/lists/*

RUN python3 -m venv /opt/venv
ENV PATH="/opt/venv/bin:${PATH}" \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

RUN pip install --upgrade pip \
    && pip install torch torchvision torchaudio \
         --index-url https://download.pytorch.org/whl/xpu

RUN git clone --depth 1 --branch "${COMFYUI_REF}" \
        https://github.com/Comfy-Org/ComfyUI.git /opt/ComfyUI \
    && pip install -r /opt/ComfyUI/requirements.txt \
    && pip install -r /opt/ComfyUI/manager_requirements.txt

# Runtime dependencies for commonly used custom nodes.
#
# ComfyUI-GGUF:
#   gguf, sentencepiece, protobuf
#
# VideoHelperSuite:
#   OpenCV, imageio-ffmpeg
#
# ffmpeg itself is installed above from Ubuntu.
RUN pip install \
        "gguf>=0.13.0" \
        sentencepiece \
        protobuf \
        opencv-python-headless \
        imageio-ffmpeg

WORKDIR /opt/ComfyUI
EXPOSE 8188

CMD ["python", "main.py", "--listen", "0.0.0.0", "--port", "8188", "--enable-manager", "--enable-manager-legacy-ui"]
