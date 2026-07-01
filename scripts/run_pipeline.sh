#!/bin/bash
set -e

cd "$(dirname "$0")/.."

echo "========================================================"
echo " Agentic RL Training Pipeline"
echo "========================================================"
echo ""

python -m main --pipeline --samples 300
