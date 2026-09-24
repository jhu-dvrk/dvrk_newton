#!/usr/bin/env bash
set -euo pipefail

# Bootstrap a dedicated Python virtualenv for NVIDIA Newton with Warp/CUDA support.
WORKSPACE_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
VENV_DIR="${WORKSPACE_ROOT}/.venv-newton"
PYTHON="${PYTHON:-python3}"

echo "Bootstrapping Newton virtual environment at: ${VENV_DIR}"
if [ ! -d "${VENV_DIR}" ]; then
    "${PYTHON}" -m venv --system-site-packages "${VENV_DIR}"
fi

"${VENV_DIR}/bin/pip" install --upgrade pip
"${VENV_DIR}/bin/pip" install \
    "newton==1.6.0" \
    "warp-lang>=1.17.0" \
    "numpy" \
    "scipy" \
    "trimesh" \
    "pycollada" \
    "pyglet>=2.0" \
    "pillow" \
    "imgui-bundle" \
    "PyYAML"

echo "Newton environment bootstrapped successfully."
echo "Test with: ${VENV_DIR}/bin/python3 -c 'import newton; print(\"Newton ready!\")'"
