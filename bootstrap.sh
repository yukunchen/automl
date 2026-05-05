#!/usr/bin/env bash
# First-time setup on a fresh box (e.g. AutoDL 5090 image).
# Idempotent — safe to re-run.
#
# Usage:
#   QAI_HUB_TOKEN=<your-token> bash bootstrap.sh
#
# Steps:
#   1. setup_train.sh (venvs + smoke tests)
#   2. configure ~/.qai_hub/client.ini (from env var)
#   3. unpack COCO val + annotations into the cache (uses AutoDL preloaded
#      zips when available)
#   4. cache the teacher checkpoint (so first student.py run doesn't
#      block on torchvision CDN)
#
# To also pull COCO train2017, pass FULL_COCO=1.

set -euo pipefail
export LANG=C

REPO_ROOT="$(cd "$(dirname "$0")" && pwd)"
cd "$REPO_ROOT"

echo "[bootstrap] running setup_train.sh"
bash setup_train.sh

echo "[bootstrap] configuring AI Hub"
if [ -z "${QAI_HUB_TOKEN:-}" ]; then
    if [ -f ~/.qai_hub/client.ini ] && grep -q "^api_token" ~/.qai_hub/client.ini; then
        echo "  ~/.qai_hub/client.ini already configured, skipping"
    else
        echo "  WARNING: QAI_HUB_TOKEN env var not set and no client.ini exists."
        echo "  Run:    export QAI_HUB_TOKEN=<your-token> && bash bootstrap.sh"
        echo "  Or set ~/.qai_hub/client.ini manually."
    fi
else
    mkdir -p ~/.qai_hub
    cat > ~/.qai_hub/client.ini <<EOF
[api]
api_token = $QAI_HUB_TOKEN
api_url = https://app.aihub.qualcomm.com
web_url = https://app.aihub.qualcomm.com
EOF
    chmod 600 ~/.qai_hub/client.ini
    echo "  ~/.qai_hub/client.ini written"
fi

echo "[bootstrap] preparing COCO"
if [ -f /root/miniconda3/etc/profile.d/conda.sh ]; then
    # shellcheck disable=SC1091
    source /root/miniconda3/etc/profile.d/conda.sh
    conda activate base
fi
# shellcheck disable=SC1091
source .venv-train/bin/activate
COCO_ARGS="--download"
if [ "${FULL_COCO:-0}" = "1" ]; then
    COCO_ARGS="$COCO_ARGS --full"
fi
python prepare.py $COCO_ARGS
deactivate

echo "[bootstrap] caching teacher checkpoint"
# shellcheck disable=SC1091
source .venv-train/bin/activate
python prepare.py --cache-teacher || echo "  (teacher cache will be lazy-loaded on first run)"
deactivate

echo "[bootstrap] done. Quick sanity:"
ls -lah ~/.cache/autoresearch-cv/coco/val2017/ 2>/dev/null | head -3 || true
echo
echo "Next step: run a baseline experiment"
echo "    bash run_experiment.sh \"baseline\""
