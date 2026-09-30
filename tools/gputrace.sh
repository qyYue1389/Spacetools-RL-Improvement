#!/bin/bash
while true; do
  echo "$(date +%s) $(nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits | paste -sd'|')"
  sleep 1
done
