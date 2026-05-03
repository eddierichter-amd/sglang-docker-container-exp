#!/usr/bin/env bash
set -euo pipefail

IMAGE_TAG="${IMAGE_TAG:-local/sglang:v0.5.9-rocm720-mi300x-smoke-py312}"
HF_CACHE="${HF_CACHE:-/scratch/eddier/hf-cache}"
AITER_CACHE="${AITER_CACHE:-/scratch/eddier/aiter-cache}"
BASE_PORT="${BASE_PORT:-31010}"
MEM_FRACTION="${MEM_FRACTION:-0.7}"
PASS_TIMEOUT_SECS="${PASS_TIMEOUT_SECS:-1800}"
FAIL_TIMEOUT_SECS="${FAIL_TIMEOUT_SECS:-900}"
RESULTS_FILE="${RESULTS_FILE:-/scratch/eddier/sglang-docker-container-exp/mi300x-smoke-matrix-results.tsv}"

mkdir -p "${HF_CACHE}" "${AITER_CACHE}"

cases=(
  "qwen25_default|Qwen/Qwen2.5-0.5B-Instruct|0|pass||"
  "qwen25_bf16|Qwen/Qwen2.5-0.5B-Instruct|0|pass|--dtype bfloat16|"
  "qwen25_tp1|Qwen/Qwen2.5-0.5B-Instruct|0|pass|--tensor-parallel-size 1|"
  "qwen3_default|Qwen/Qwen3-4B-Instruct-2507|0|pass||"
  "qwen3_tp8|Qwen/Qwen3-4B-Instruct-2507|0,1,2,3,4,5,6,7|pass|--tensor-parallel-size 8|"
  "qwen3_fp16|Qwen/Qwen3-4B-Instruct-2507|0|pass|--dtype float16|"
  "qwen3_tp8_fp32|Qwen/Qwen3-4B-Instruct-2507|0,1,2,3,4,5,6,7|fail|--tensor-parallel-size 8 --dtype float32|"
  "qwen3fp8_tp1|Qwen/Qwen3-4B-FP8|0|pass|--tensor-parallel-size 1|"
  "qwen3fp8_tp2|Qwen/Qwen3-4B-FP8|0,1|pass|--tensor-parallel-size 2|"
  "qwen3fp8_tp4|Qwen/Qwen3-4B-FP8|0,1,2,3|pass|--tensor-parallel-size 4|"
  "qwen3fp8_tp8|Qwen/Qwen3-4B-FP8|0,1,2,3,4,5,6,7|fail|--tensor-parallel-size 8|FP8 block partitioning mismatch"
)

printf 'case\texpected\tstatus\tport\tmodel\tvisible_devices\textra_args\tcompletion\tlog_hint\n' > "${RESULTS_FILE}"

cleanup_container() {
  local container_name="$1"
  docker rm -f "${container_name}" >/dev/null 2>&1 || true
}

wait_for_pass_ready() {
  local container_name="$1"
  local port="$2"
  local timeout_secs="$3"
  local deadline=$((SECONDS + timeout_secs))

  while (( SECONDS < deadline )); do
    if curl -fsS "http://127.0.0.1:${port}/v1/models" >/tmp/"${container_name}".models.json 2>/dev/null; then
      return 0
    fi
    if ! docker ps --format '{{.Names}}' | grep -qx "${container_name}"; then
      return 1
    fi
    sleep 5
  done

  return 2
}

wait_for_expected_failure() {
  local container_name="$1"
  local port="$2"
  local timeout_secs="$3"
  local log_hint="$4"
  local deadline=$((SECONDS + timeout_secs))

  while (( SECONDS < deadline )); do
    if curl -fsS "http://127.0.0.1:${port}/v1/models" >/dev/null 2>&1; then
      return 10
    fi

    if ! docker ps --format '{{.Names}}' | grep -qx "${container_name}"; then
      return 0
    fi

    if [[ -n "${log_hint}" ]] && docker logs "${container_name}" 2>&1 | grep -Fq "${log_hint}"; then
      return 0
    fi

    sleep 5
  done

  return 1
}

extract_completion_text() {
  local completion_file="$1"

  if command -v jq >/dev/null 2>&1; then
    jq -r '(.choices[0].message.content // "") | gsub("\r"; "") | gsub("\n"; "\\n")' "${completion_file}"
  else
    python3 - "${completion_file}" <<'PY'
import json
import sys

with open(sys.argv[1], "r", encoding="utf-8") as f:
    data = json.load(f)

content = data.get("choices", [{}])[0].get("message", {}).get("content", "")
content = content.replace("\r", "").replace("\n", "\\n")
print(content)
PY
  fi
}

completion_is_chat_object() {
  local completion_file="$1"

  if command -v jq >/dev/null 2>&1; then
    jq -e '.object == "chat.completion"' "${completion_file}" >/dev/null
  else
    python3 - "${completion_file}" <<'PY'
import json
import sys

with open(sys.argv[1], "r", encoding="utf-8") as f:
    data = json.load(f)

raise SystemExit(0 if data.get("object") == "chat.completion" else 1)
PY
  fi
}

run_case() {
  local index="$1"
  local case_name="$2"
  local model_path="$3"
  local visible_devices="$4"
  local expected="$5"
  local extra_args="$6"
  local log_hint="$7"

  local port=$((BASE_PORT + index))
  local container_name="sglang-matrix-${case_name}"
  local served_name
  local launch_cmd
  local log_file="/tmp/${container_name}.log"
  local completion=""
  local status=""

  served_name="${model_path##*/}"
  cleanup_container "${container_name}"
  rm -f "${log_file}" "/tmp/${container_name}.models.json" "/tmp/${container_name}.completion.json"

  launch_cmd="python -m sglang.launch_server --model-path ${model_path} --host 0.0.0.0 --port ${port} --mem-fraction-static ${MEM_FRACTION} --served-model-name ${served_name}"
  if [[ -n "${extra_args}" ]]; then
    launch_cmd+=" ${extra_args}"
  fi

  echo "==> ${case_name} (${expected}) on port ${port}"

  docker run -d \
    --name "${container_name}" \
    --network host \
    --device=/dev/kfd \
    --device=/dev/dri \
    --group-add video \
    --group-add render \
    --ipc=host \
    --shm-size 16g \
    -e ROCR_VISIBLE_DEVICES="${visible_devices}" \
    -e HIP_VISIBLE_DEVICES="${visible_devices}" \
    -e CUDA_VISIBLE_DEVICES="${visible_devices}" \
    -v "${HF_CACHE}:/root/.cache/huggingface" \
    -v "${AITER_CACHE}:/root/.aiter" \
    "${IMAGE_TAG}" \
    /bin/bash -lc "${launch_cmd}" >/dev/null

  if [[ "${expected}" == "pass" ]]; then
    if wait_for_pass_ready "${container_name}" "${port}" "${PASS_TIMEOUT_SECS}"; then
      if curl -fsS "http://127.0.0.1:${port}/v1/chat/completions" \
        -H 'Content-Type: application/json' \
        -d "{\"model\":\"${served_name}\",\"messages\":[{\"role\":\"user\",\"content\":\"Reply with exactly: OK\"}],\"temperature\":0,\"max_tokens\":8}" \
        >"/tmp/${container_name}.completion.json"; then
        completion="$(extract_completion_text "/tmp/${container_name}.completion.json" | head -n 1)"
        if completion_is_chat_object "/tmp/${container_name}.completion.json"; then
          status="PASS"
        else
          status="FAIL_COMPLETION"
        fi
      else
        status="FAIL_COMPLETION_REQUEST"
      fi
    else
      case "$?" in
        1) status="FAIL_EXITED_BEFORE_READY" ;;
        2) status="FAIL_TIMEOUT" ;;
        *) status="FAIL_READY_CHECK" ;;
      esac
    fi
  else
    if wait_for_expected_failure "${container_name}" "${port}" "${FAIL_TIMEOUT_SECS}" "${log_hint}"; then
      status="EXPECTED_FAIL"
    else
      case "$?" in
        1) status="UNRESOLVED_TIMEOUT" ;;
        10) status="UNEXPECTED_PASS" ;;
        *) status="UNEXPECTED_FAIL_STATE" ;;
      esac
    fi
  fi

  docker logs --tail 400 "${container_name}" >"${log_file}" 2>&1 || true
  printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
    "${case_name}" \
    "${expected}" \
    "${status}" \
    "${port}" \
    "${model_path}" \
    "${visible_devices}" \
    "${extra_args}" \
    "${completion}" \
    "${log_hint}" \
    >> "${RESULTS_FILE}"

  cleanup_container "${container_name}"
}

for i in "${!cases[@]}"; do
  IFS='|' read -r case_name model_path visible_devices expected extra_args log_hint <<<"${cases[$i]}"
  run_case "${i}" "${case_name}" "${model_path}" "${visible_devices}" "${expected}" "${extra_args}" "${log_hint}"
done

echo
echo "Results written to ${RESULTS_FILE}"
column -t -s $'\t' "${RESULTS_FILE}"
