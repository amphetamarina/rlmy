#!/usr/bin/env bash
# Build + run rlmy in Docker, passing AWS creds from your host environment.
# Usage: bash docker-run.sh
set -euo pipefail

IMAGE="rlmy:local"

echo "=== Building ${IMAGE} ==="
docker build -t "${IMAGE}" .

echo "=== Running rlmy (interactive) ==="
# -it           : interactive TTY (rlmy uses prompt_toolkit)
# -e AWS_*      : inherit AWS creds from your host shell (no secrets baked into image)
# -e RLM_*      : skip the wizard by pre-selecting Bedrock models
# -v rlmy-data  : persist sandboxes/trajectories across runs
docker run -it --rm \
    -e AWS_ACCESS_KEY_ID \
    -e AWS_SECRET_ACCESS_KEY \
    -e AWS_SESSION_TOKEN \
    -e AWS_REGION="${AWS_REGION:-us-west-2}" \
    -e RLM_MAIN_MODEL="${RLM_MAIN_MODEL:-bedrock/us.anthropic.claude-sonnet-4-6}" \
    -e RLM_SUB_MODEL="${RLM_SUB_MODEL:-bedrock/us.anthropic.claude-sonnet-4-6}" \
    -v rlmy-data:/root/.config/rlmy/sandboxes \
    "${IMAGE}"
