#!/usr/bin/env bash
# Start Ollama with settings this project needs (see .plan/gotcha.md).
#   OLLAMA_CONTEXT_LENGTH  default 4096 silently truncates agent prompts; 8K fits ours and
#                          keeps the KV cache small on 16 GB
#   OLLAMA_FLASH_ATTENTION + OLLAMA_KV_CACHE_TYPE=q8_0  roughly halve KV-cache memory
export OLLAMA_CONTEXT_LENGTH="${OLLAMA_CONTEXT_LENGTH:-8192}"
export OLLAMA_FLASH_ATTENTION=1
export OLLAMA_KV_CACHE_TYPE=q8_0
export OLLAMA_MAX_LOADED_MODELS=2 # chat + embedding model; avoid loading more on 16 GB
exec ollama serve
