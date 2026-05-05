#!/usr/bin/env bash
# One-shot setup for the 4090/5090 training box.
# Run from the repo root after `git clone && git checkout autoresearch/cv-detection`.
#
# Assumes the AutoDL-style image: Ubuntu 22.04, CUDA 12.8, PyTorch 2.7 preinstalled,
# Python 3.12 system. Creates two venvs:
#   .venv-train    - training stack (Python 3.12, uses preinstalled torch)
#   .venv-deploy   - deployment stack (Python 3.11, qai-hub + onnx)

set -euo pipefail
export LANG=C

# AutoDL ships PyTorch inside a conda env that isn't active in non-interactive
# shells. Source it if present.
if [ -f /root/miniconda3/etc/profile.d/conda.sh ]; then
    # shellcheck disable=SC1091
    source /root/miniconda3/etc/profile.d/conda.sh
    conda activate base
fi

PY=$(command -v python || command -v python3)
echo "using python: $PY"

echo "[1/6] sanity-checking GPU + driver"
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv
"$PY" -c "import torch; print('torch', torch.__version__, 'cuda', torch.cuda.is_available(), torch.cuda.get_device_name(0))"

echo "[2/6] BF16 + CUDA matmul smoke test"
"$PY" -c "import torch; x=torch.randn(1024,1024,device='cuda',dtype=torch.bfloat16); y=(x@x).sum().item(); print(f'bf16 matmul ok, sum={y:.2f}')"

echo "[3/6] installing uv (if missing)"
if ! command -v uv >/dev/null 2>&1; then
    curl -LsSf https://astral.sh/uv/install.sh | sh
    export PATH="$HOME/.local/bin:$PATH"
fi
uv --version

echo "[4/6] training venv (.venv-train, system Python, uses preinstalled torch)"
if [ ! -d .venv-train ]; then
    "$PY" -m venv --system-site-packages .venv-train
fi
# shellcheck disable=SC1091
source .venv-train/bin/activate
pip install --quiet --upgrade pip
pip install --quiet timm pycocotools opencv-python-headless tqdm tensorboard
python -c "import torch, timm, pycocotools; print('train venv ok')"
deactivate

echo "[5/6] deploy venv (.venv-deploy, Python 3.11 via uv, qai-hub stack)"
# Use Tsinghua mirror for China-hosted boxes; harmless elsewhere.
# Cache wheels on the data disk (system disk on AutoDL is only ~30GB).
export UV_CACHE_DIR="${UV_CACHE_DIR:-$PWD/.uv-cache}"
PIP_INDEX="${PIP_INDEX_URL:-https://pypi.tuna.tsinghua.edu.cn/simple}"
if [ ! -d .venv-deploy ]; then
    uv venv --python 3.11 .venv-deploy
fi
# shellcheck disable=SC1091
source .venv-deploy/bin/activate
uv pip install --quiet --index-url "$PIP_INDEX" qai-hub torch torchvision onnx onnxruntime
python -c "import qai_hub, torch, onnx; print('deploy venv ok, qai_hub', qai_hub.__version__)"
deactivate

echo "[6/6] dataset cache dir"
mkdir -p ~/.cache/autoresearch-cv

echo
echo "Done. Next steps:"
echo "  1. Configure qai-hub:    mkdir -p ~/.qai_hub && \$EDITOR ~/.qai_hub/client.ini"
echo "     (paste api_token from AI Hub Settings; do NOT commit)"
echo "  2. Download COCO into ~/.cache/autoresearch-cv/coco/  (when prepare.py is ready)"
echo "  3. Activate train venv:  source .venv-train/bin/activate"
