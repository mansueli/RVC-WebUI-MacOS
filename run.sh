#!/usr/bin/env bash

set -fa

required_python="3.10"

find_python310() {
  if command -v python3.10 >/dev/null 2>&1; then
    echo "python3.10"
    return
  fi
  if command -v python >/dev/null 2>&1; then
    pyver="$(python -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")' 2>/dev/null)"
    if [[ "${pyver}" == "${required_python}" ]]; then
      echo "python"
      return
    fi
  fi
  echo ""
}

python_cmd="$(find_python310)"

if [[ -z "${python_cmd}" ]]; then
  echo "Python ${required_python}.x is required."
  echo "Install Python ${required_python} and re-run this script."
  exit 1
fi

requirements_file="requirements/python310.txt"
venv_path=".venv"

if [[ ! -d "${venv_path}" ]]; then
  echo "Creating venv..."

  "${python_cmd}" -m venv "${venv_path}"
  source "${venv_path}/bin/activate"

  # Check if required packages are up-to-date
  pip install --upgrade -r "${requirements_file}"
fi
echo "Activating venv..."
source "${venv_path}/bin/activate"

venv_python_version="$(python -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')"
if [[ "${venv_python_version}" != "${required_python}" ]]; then
  echo "Current virtual environment uses Python ${venv_python_version}."
  echo "Please recreate .venv with Python ${required_python}.x:"
  echo "  rm -rf .venv && ${python_cmd} -m venv .venv"
  exit 1
fi

# Optional: download/check required assets before launching WebUI.
# Enable with: AUTO_DOWNLOAD_ASSETS=1 ./run.sh
if [[ "${AUTO_DOWNLOAD_ASSETS:-0}" == "1" ]]; then
  echo "Checking and downloading missing assets..."
  python tools/download_models.py
  if [[ $? -ne 0 ]]; then
    echo "Asset bootstrap failed. Aborting startup."
    exit 1
  fi
fi

# Run the main script
python web.py --pycmd python
