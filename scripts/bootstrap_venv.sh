#!/usr/bin/env bash
set -euo pipefail

# Bootstrap a dedicated Python virtualenv for NVIDIA Newton with Warp/CUDA support.
SCRIPT_DIR="$(dirname "$(readlink -f "${BASH_SOURCE[0]}")")"

BASE_BOOTSTRAP="${SCRIPT_DIR}/../../dvrk_simulator_base/scripts/bootstrap_venv.sh"
if [[ ! -f "${BASE_BOOTSTRAP}" ]]; then
    BASE_PREFIX="$(ros2 pkg prefix dvrk_simulator_base 2>/dev/null || true)"
    if [[ -n "${BASE_PREFIX}" && -f "${BASE_PREFIX}/share/dvrk_simulator_base/scripts/bootstrap_venv.sh" ]]; then
        BASE_BOOTSTRAP="${BASE_PREFIX}/share/dvrk_simulator_base/scripts/bootstrap_venv.sh"
    fi
fi

if [[ ! -f "${BASE_BOOTSTRAP}" ]]; then
    echo "error: bootstrap_venv.sh from dvrk_simulator_base not found" >&2
    exit 2
fi

exec "${BASE_BOOTSTRAP}" \
    ".venv-newton" \
    "${SCRIPT_DIR}/../requirements.txt" \
    "NVIDIA Newton" \
    "-c 'import newton; print(\"Newton ready!\")'" \
    "$@"
