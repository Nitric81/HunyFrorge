# syntax=docker/dockerfile:1
FROM node:22-bookworm-slim AS frontend
WORKDIR /app
COPY package.json package-lock.json ./
RUN npm ci
COPY index.html tsconfig.json vite.config.ts ./
COPY src ./src
RUN npm run build

FROM nvidia/cuda:12.4.1-cudnn-devel-ubuntu22.04
ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    HUNYFORGE_DEMO=0 \
    HUNYFORGE_DATA_ROOT=/data \
    HUNYUAN_ROOT=/opt/hunyuan \
    HUNYUAN_MODEL_PATH=/models/Hunyuan3D-2.1 \
    HUNYUAN_API_URL=http://127.0.0.1:8082 \
    HUNYUAN_REQUEST_TIMEOUT_SECONDS=3600 \
    HUNYUAN_DINO_DEVICE=cuda \
    HUNYUAN_FLASHVDM=1 \
    HUNYUAN_COMPILE=0 \
    HUNYUAN_CPU_THREADS=8 \
    HUNYFORGE_T2I_MODEL_PATH=/models/FLUX.2-klein-4B \
    HUNYFORGE_T2I_ENABLED=1 \
    HUNYFORGE_T2I_MAX_PROMPT_CHARS=2000 \
    HF_HUB_OFFLINE=1 \
    TRANSFORMERS_OFFLINE=1 \
    HF_HUB_DISABLE_TELEMETRY=1 \
    PYTHONPATH=/app
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.10 python3.10-dev python3-pip python3-setuptools \
    build-essential cmake ninja-build pkg-config git git-lfs wget \
    libcgal-dev libeigen3-dev libegl1-mesa-dev libgl1-mesa-dev \
    libgl1 libglib2.0-0 libgles2-mesa-dev libglvnd-dev libgomp1 \
    libsm6 libxext6 libxi6 libxkbcommon-x11-0 libxrender1 libxrender-dev \
    && rm -rf /var/lib/apt/lists/*
ARG HUNYUAN_SOURCE_REVISION=82920d643c0dc2f7bfd7255f45f62d386edfe60c
RUN git init /opt/hunyuan && \
    git -C /opt/hunyuan remote add origin https://github.com/Tencent-Hunyuan/Hunyuan3D-2.1.git && \
    git -C /opt/hunyuan fetch --depth 1 origin ${HUNYUAN_SOURCE_REVISION} && \
    git -C /opt/hunyuan checkout --detach FETCH_HEAD
ENV HUNYUAN_SOURCE_REVISION=${HUNYUAN_SOURCE_REVISION} \
    HUNYUAN_MODEL_REVISION=0b94677654c57bb9a6b6845cd7b704ccf551d327
# The API worker imports mesh_utils, but its production texture path passes
# save_glb=False and uses hy3dpaint.convert_utils for PBR GLB output. Make the
# otherwise-unused Blender import optional; bpy 4.0 has no CPython 3.10 Linux
# wheel. Calls to the Blender-only helper remain unavailable by design.
RUN sed -i 's/^import bpy$/try:\n    import bpy\nexcept ImportError:\n    bpy = None/' \
    /opt/hunyuan/hy3dpaint/DifferentiableRenderer/mesh_utils.py
# Keep heavyweight CUDA/PyTorch in its own cacheable layer. Rebuilding a later
# dependency must not download several gigabytes again.
RUN --mount=type=cache,target=/root/.cache/pip \
    python3.10 -m pip install --upgrade pip setuptools wheel && \
    python3.10 -m pip install \
      torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 \
      --index-url https://download.pytorch.org/whl/cu124

# bpy is only needed when running Blender's embedded Python. No bpy 4.0 wheel
# exists for CPython 3.10/Linux, and neither api_server.py nor HunyForge imports
# it. basicsr is installed without build isolation to prevent pip from creating
# a temporary environment and downloading a second copy of PyTorch.
RUN --mount=type=cache,target=/root/.cache/pip \
    grep -vE '^(--extra-index-url|bpy==4\.0|basicsr==1\.4\.2)' /opt/hunyuan/requirements.txt > /tmp/hunyuan-runtime-requirements.txt && \
    python3.10 -m pip install --no-build-isolation basicsr==1.4.2 && \
    python3.10 -m pip install -r /tmp/hunyuan-runtime-requirements.txt

# Compile the two PBR renderer extensions required by Hunyuan3D-Paint.
ENV CUDA_HOME=/usr/local/cuda \
    PYOPENGL_PLATFORM=egl \
    TORCH_CUDA_ARCH_LIST=8.9
RUN --mount=type=cache,target=/root/.cache/pip \
    python3.10 -m pip install --no-build-isolation -e /opt/hunyuan/hy3dpaint/custom_rasterizer && \
    ln -sf /usr/bin/python3.10 /usr/local/bin/python && \
    ln -sf /usr/bin/python3.10-config /usr/local/bin/python3-config && \
    cd /opt/hunyuan/hy3dpaint/DifferentiableRenderer && bash compile_mesh_painter.sh && \
    mkdir -p /opt/hunyuan/hy3dpaint/ckpt && \
    wget -q https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth \
      -O /opt/hunyuan/hy3dpaint/ckpt/RealESRGAN_x4plus.pth

# Upgrade diffusers/transformers for FLUX.2-klein-4B (Flux2KleinPipeline needs
# diffusers >=0.37; its Qwen3 text encoder needs transformers >=4.51). Kept as
# a separate layer after the CUDA extension builds so those stay cached.
# hub must stay <1.0 for transformers 4.51; safetensors >=0.8 and tokenizers
# 0.21.x are required by the new packages. deepspeed is unused at inference
# and its tp_collectives custom op fails to import on torch 2.5.1; removing
# it also lets transformers 4.51+ skip its eager import.
RUN --mount=type=cache,target=/root/.cache/pip \
    python3.10 -m pip uninstall -y deepspeed && \
    python3.10 -m pip install \
      "diffusers==0.38.0" "transformers==4.51.3" \
      "huggingface-hub==0.36.2" "safetensors==0.8.0" \
      "tokenizers==0.21.4" "accelerate==1.10.1"
COPY docker/hunyuan-local-models.patch /tmp/hunyuan-local-models.patch
RUN cd /opt/hunyuan && patch -p1 < /tmp/hunyuan-local-models.patch
COPY docker/configure_runtime.py /tmp/configure_runtime.py
RUN python3.10 /tmp/configure_runtime.py /opt/hunyuan
RUN python3.10 -c "from pathlib import Path; p=Path('/opt/hunyuan/api_models.py'); s=p.read_text(); s=s.replace('    seed: int = Field(', '    mesh: Optional[str] = Field(\n        None,\n        description=\"Optional base64 GLB mesh to texture without regenerating shape\"\n    )\n    seed: int = Field(', 1); p.write_text(s)"
RUN python3.10 -c "from pathlib import Path; p=Path('/opt/hunyuan/api_server.py'); s=p.read_text(); s=s.replace('\"text\": SERVER_ERROR_MSG,', '\"text\": str(e),', 1); s=s.replace('return JSONResponse(ret, status_code=404)', 'return JSONResponse(ret, status_code=500)', 1); p.write_text(s)"
WORKDIR /app
COPY backend ./backend
COPY --from=frontend /app/dist ./dist
COPY docker/entrypoint.sh ./docker/entrypoint.sh
RUN chmod +x docker/entrypoint.sh && mkdir -p /data /models
EXPOSE 8081
ENTRYPOINT ["/app/docker/entrypoint.sh"]
