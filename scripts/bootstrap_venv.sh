#!/usr/bin/env bash
set -euo pipefail

# Bootstrap a dedicated Python virtualenv for NVIDIA Newton with Warp/CUDA support.
SCRIPT_PATH="$(readlink -f "${BASH_SOURCE[0]}")"
SCRIPT_DIR="$(dirname "${SCRIPT_PATH}")"

find_workspace_root() {
    local cursor="${SCRIPT_DIR}"
    while [[ "${cursor}" != "/" ]]; do
        case "$(basename "${cursor}")" in
            src|install)
                dirname "${cursor}"
                return 0
                ;;
        esac
        cursor="$(dirname "${cursor}")"
    done
    return 1
}

WORKSPACE_ROOT="$(find_workspace_root || true)"
if [[ -z "${WORKSPACE_ROOT}" ]]; then
    echo "error: could not determine workspace root from ${SCRIPT_PATH}" >&2
    exit 2
fi

VENV_DIR="${WORKSPACE_ROOT}/.venv-newton"
REQUIREMENTS_FILE="${SCRIPT_DIR}/../requirements.txt"
if [[ ! -f "${REQUIREMENTS_FILE}" ]]; then
    REQUIREMENTS_FILE="${SCRIPT_DIR}/requirements.txt"
fi

if [[ ! -f "${REQUIREMENTS_FILE}" ]]; then
    echo "error: requirements.txt not found" >&2
    exit 2
fi
REQUIREMENTS_FILE="$(readlink -f "${REQUIREMENTS_FILE}")"
PYTHON="${PYTHON:-python3}"

echo "Bootstrapping Newton virtual environment at: ${VENV_DIR}"
echo "Requirements file: ${REQUIREMENTS_FILE}"

# Check for non-interactive flag (-y / --yes)
PROCEED=false
for arg in "$@"; do
    case "$arg" in
        -y|--yes)
            PROCEED=true
            ;;
    esac
done

if [ "${PROCEED}" = false ]; then
    if [ -t 0 ]; then
        read -r -p "Create venv and install using pip? [y/N] " response
        case "${response}" in
            [yY][eE][sS]|[yY])
                ;;
            *)
                echo "Operation cancelled."
                exit 0
                ;;
        esac
    fi
fi

if [ ! -d "${VENV_DIR}" ]; then
    echo "Creating virtual environment at ${VENV_DIR}..."
    "${PYTHON}" -m venv --system-site-packages "${VENV_DIR}"
fi

echo "Installing dependencies from ${REQUIREMENTS_FILE}..."
"${VENV_DIR}/bin/pip" install --upgrade pip
"${VENV_DIR}/bin/pip" install -r "${REQUIREMENTS_FILE}"

echo "Newton environment bootstrapped successfully."
echo "Test with: ${VENV_DIR}/bin/python3 -c 'import newton; print(\"Newton ready!\")'"
