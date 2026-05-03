#!/usr/bin/env bash
set -euo pipefail

IMAGE_TAG="${IMAGE_TAG:-local/sglang:v0.5.9-rocm720-mi300x-smoke-py312}"
SGL_BRANCH="${SGL_BRANCH:-v0.5.9}"
GPU_ARCH="${GPU_ARCH:-gfx942-rocm720-smoke}"
BUILD_TYPE="${BUILD_TYPE:-srt}"
ENABLE_MORI="${ENABLE_MORI:-0}"
NIC_BACKEND="${NIC_BACKEND:-none}"
UBUNTU_CODENAME="${UBUNTU_CODENAME:-noble}"
BASE_IMAGE_942_ROCM720="${BASE_IMAGE_942_ROCM720:-rocm/pytorch:rocm7.2_ubuntu24.04_py3.12_pytorch_release_2.9.1}"
BASE_IMAGE_950_ROCM720="${BASE_IMAGE_950_ROCM720:-$BASE_IMAGE_942_ROCM720}"

if docker buildx version >/dev/null 2>&1; then
  export DOCKER_BUILDKIT="${DOCKER_BUILDKIT:-1}"
else
  export DOCKER_BUILDKIT="${DOCKER_BUILDKIT:-0}"
fi

docker build \
  --build-arg SGL_BRANCH="${SGL_BRANCH}" \
  --build-arg GPU_ARCH="${GPU_ARCH}" \
  --build-arg BUILD_TYPE="${BUILD_TYPE}" \
  --build-arg ENABLE_MORI="${ENABLE_MORI}" \
  --build-arg NIC_BACKEND="${NIC_BACKEND}" \
  --build-arg UBUNTU_CODENAME="${UBUNTU_CODENAME}" \
  --build-arg BASE_IMAGE_942_ROCM720="${BASE_IMAGE_942_ROCM720}" \
  --build-arg BASE_IMAGE_950_ROCM720="${BASE_IMAGE_950_ROCM720}" \
  -t "${IMAGE_TAG}" \
  -f docker/rocm720.Dockerfile \
  docker
