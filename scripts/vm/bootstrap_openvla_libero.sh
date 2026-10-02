#!/usr/bin/env bash
# Bootstrap the pinned *local* runtime. Run on an NVIDIA Linux VM, not macOS.
set -euo pipefail

if [[ -z "${VLA_XRT_ROOT:-}" ]]; then
  echo "Set VLA_XRT_ROOT to the absolute path of this checkout." >&2
  exit 2
fi
if [[ ! -f "$VLA_XRT_ROOT/pyproject.toml" ]]; then
  echo "VLA_XRT_ROOT does not look like a VLA-XRT checkout: $VLA_XRT_ROOT" >&2
  exit 2
fi

VLA_XRT_ENV="${VLA_XRT_ENV:-openvla-xrt}"
OPENVLA_REF="${OPENVLA_REF:-main}"
LIBERO_REF="${LIBERO_REF:-master}"
UPSTREAM_DIR="$VLA_XRT_ROOT/third_party"
OPENVLA_DIR="$UPSTREAM_DIR/openvla"
LIBERO_DIR="$UPSTREAM_DIR/LIBERO"

command -v conda >/dev/null || { echo "conda is required; install Miniforge first." >&2; exit 2; }
command -v git >/dev/null || { echo "git is required." >&2; exit 2; }

conda create --yes --name "$VLA_XRT_ENV" python=3.10
eval "$(conda shell.bash hook)"
conda activate "$VLA_XRT_ENV"

# Install the PyTorch CUDA wheel appropriate for the VM before this script, if
# needed. These are the versions tested by the upstream OpenVLA project.
python -m pip install --upgrade pip packaging ninja
TORCH_INSTALL=(python -m pip install)
if [[ -n "${PYTORCH_INDEX_URL:-}" ]]; then
  TORCH_INSTALL+=(--index-url "$PYTORCH_INDEX_URL")
fi
"${TORCH_INSTALL[@]}" torch==2.2.0 torchvision==0.17.0 transformers==4.40.1 tokenizers==0.19.1 timm==0.9.10

mkdir -p "$UPSTREAM_DIR" "$VLA_XRT_ROOT/.vla-xrt"
if [[ ! -d "$OPENVLA_DIR/.git" ]]; then git clone https://github.com/openvla/openvla.git "$OPENVLA_DIR"; fi
if [[ ! -d "$LIBERO_DIR/.git" ]]; then git clone https://github.com/Lifelong-Robot-Learning/LIBERO.git "$LIBERO_DIR"; fi
git -C "$OPENVLA_DIR" fetch --tags origin
git -C "$OPENVLA_DIR" checkout "$OPENVLA_REF"
git -C "$LIBERO_DIR" fetch --tags origin
git -C "$LIBERO_DIR" checkout "$LIBERO_REF"

python -m pip install -e "$LIBERO_DIR"
python -m pip install -e "$OPENVLA_DIR"
python -m pip install flash-attn==2.5.5 --no-build-isolation
python -m pip install -e "$VLA_XRT_ROOT"

python - <<PY
import json, subprocess
from pathlib import Path
roots = {"openvla": Path("$OPENVLA_DIR"), "libero": Path("$LIBERO_DIR")}
lock = {name: subprocess.check_output(["git", "-C", str(root), "rev-parse", "HEAD"], text=True).strip() for name, root in roots.items()}
lock["environment"] = "$VLA_XRT_ENV"
Path("$VLA_XRT_ROOT/.vla-xrt/upstream-lock.json").write_text(json.dumps(lock, indent=2) + "\\n")
print(json.dumps(lock, indent=2))
PY

echo "Bootstrap complete. Activate with: conda activate $VLA_XRT_ENV"
