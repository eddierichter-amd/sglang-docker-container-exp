# SGLang ROCm 7.2 MI35x Mori Python 3.12 Build

This repository contains the build recipe for a Python 3.12 variant of the
SGLang ROCm 7.2 MI35x Mori image line.

The built image was tested on an MI355X host and verified with:

- `Qwen/Qwen2.5-0.5B-Instruct`
- `Qwen/Qwen3-4B-Instruct-2507`
- `Qwen/Qwen3-4B-FP8`

## Files

- `build-mi35x-rocm720-py312.sh`
  - Wrapper script for `docker build`
- `docker/rocm720.Dockerfile`
  - Modified ROCm 7.2 Dockerfile
- `docker/patch_sgl_model_gateway.py`
  - Compatibility patch for `sgl-model-gateway`
- `docker/patch_torch_triton_metadata.py`
  - Torch metadata patch to relax the Triton requirement

## What Changed

Relative to the vendored upstream `v0.5.9` ROCm 7.2 Dockerfile, this build:

- switches the ROCm 7.2 base from Ubuntu 22.04 / Python 3.10 to Ubuntu 24.04 / Python 3.12
- pins SGLang checkout to `v0.5.9` instead of `main`
- switches the AMD AINIC package codename from `jammy` to `noble`
- patches `sgl-model-gateway` for the `smg-wasm 1.0.1` API shape
- patches installed Torch metadata so custom Triton `3.6.0` does not conflict with the base-image pin
- bakes `PYTHONPATH=/sgl-workspace/aiter:/sgl-workspace/mori/python:` into the image so AITER works without extra runtime overrides

## Prerequisites

- Docker with permission to access the local daemon
- ROCm device nodes available on the host:
  - `/dev/kfd`
  - `/dev/dri`
- Internet access during build for Git repositories and package downloads

## Build

Build the image with the provided wrapper:

```bash
./build-mi35x-rocm720-py312.sh
```

The default output tag is:

```text
local/sglang:v0.5.9-rocm720-mi35x-mori-py312
```

You can override the tag:

```bash
IMAGE_TAG=custom/image:tag \
./build-mi35x-rocm720-py312.sh
```

## Run

Example single-GPU launch:

```bash
docker run -d --name sglang-qwen \
  --network host \
  --device=/dev/kfd \
  --device=/dev/dri \
  --group-add video \
  --group-add render \
  --ipc=host \
  --shm-size 16g \
  -e ROCR_VISIBLE_DEVICES=0 \
  -e HIP_VISIBLE_DEVICES=0 \
  -e CUDA_VISIBLE_DEVICES=0 \
  -v /path/to/hf-cache:/root/.cache/huggingface \
  local/sglang:v0.5.9-rocm720-mi35x-mori-py312 \
  /bin/bash -lc 'python -m sglang.launch_server \
    --model-path Qwen/Qwen2.5-0.5B-Instruct \
    --host 0.0.0.0 \
    --port 31000 \
    --mem-fraction-static 0.7 \
    --served-model-name Qwen2.5-0.5B-Instruct'
```

Example health check:

```bash
curl -sS http://127.0.0.1:31000/v1/models
```

Example completion request:

```bash
curl -sS http://127.0.0.1:31000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"Qwen2.5-0.5B-Instruct","messages":[{"role":"user","content":"Reply with exactly: OK"}],"temperature":0,"max_tokens":8}'
```

## Verified Behavior

- Default attention backend selected by SGLang is `aiter`
- `Qwen/Qwen2.5-0.5B-Instruct` worked:
  - default launch
  - explicit `--dtype bfloat16`
  - explicit `--tensor-parallel-size 1`
- `Qwen/Qwen3-4B-Instruct-2507` worked:
  - default launch
  - `--tensor-parallel-size 8`
  - `--dtype float16`
- `Qwen/Qwen3-4B-Instruct-2507` did not work with:
  - `--tensor-parallel-size 8 --dtype float32`
- `Qwen/Qwen3-4B-FP8` worked with:
  - `--tensor-parallel-size 1`
  - `--tensor-parallel-size 2`
  - `--tensor-parallel-size 4`
- `Qwen/Qwen3-4B-FP8` did not work with:
  - `--tensor-parallel-size 8`
  - failure: FP8 block partitioning mismatch

## Notes

- This image is a build-oriented image, not a slim runtime-only image.
- It contains the checked-out source trees under `/sgl-workspace`.
- Hugging Face model downloads used during testing were bind-mounted at runtime and are not baked into the image.
