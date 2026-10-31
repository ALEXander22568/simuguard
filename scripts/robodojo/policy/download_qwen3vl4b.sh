#!/usr/bin/env bash
# Base Qwen3-VL-4B-Instruct weights (ModelScope Qwen/Qwen3-VL-4B-Instruct) next to the processor files.
cd "$(dirname "$0")/Qwen3-VL-4B-Instruct-processor"
for f in model.safetensors.index.json model-00001-of-00002.safetensors model-00002-of-00002.safetensors; do
  for i in 1 2 3 4 5; do
    curl -s -L -C - --retry 5 -o $f "https://modelscope.cn/api/v1/models/Qwen/Qwen3-VL-4B-Instruct/repo?Revision=master&FilePath=$f" && break
    sleep 10
  done
done
ls -la; echo DONE $(date -u +%FT%TZ)
