#!/usr/bin/env bash
# Start Ollama with settings this project needs (see .plan/gotcha.md).
#   OLLAMA_CONTEXT_LENGTH  default is 4096; longer prompts get silently truncated
#   OLLAMA_FLASH_ATTENTION + OLLAMA_KV_CACHE_TYPE=q8_0  roughly halve KV-cache memory
export OLLAMA_CONTEXT_LENGTH="${OLLAMA_CONTEXT_LENGTH:-16384}"
export OLLAMA_FLASH_ATTENTION=1
export OLLAMA_KV_CACHE_TYPE=q8_0
export OLLAMA_MAX_LOADED_MODELS=2 # chat + embedding model; avoid loading more on 16 GB
exec ollama serve
