#!/usr/bin/env bash
set -euo pipefail

# Helper script to run the notebook inside the project's .venv and capture logs.
# Usage: bash run_in_tmux.sh

SESSION_NAME="if_train"
NOTEBOOK="if_statement_predictor.ipynb"
LOGFILE="run_training.log"
EXECUTED_NOTEBOOK="executed-${NOTEBOOK}"

# Use the virtualenv's python if available
if [ -x ".venv/bin/python" ]; then
  PYTHON=".venv/bin/python"
else
  PYTHON=python
fi

# Make sure working dir is the script dir (project root)
cd "$(dirname "$0")"

echo "Starting notebook execution: $(date)" | tee -a "$LOGFILE"

# (Optional) install requirements if you want to ensure all packages are present
# Uncomment the next line if you want automatic installs (takes time):
# $PYTHON -m pip install -r requirements.txt 2>&1 | tee -a "$LOGFILE"

# Execute the notebook (unbuffered output via stdbuf if available) and append to log
# Use python -m jupyter.nbconvert so that the venv's jupyter is used
$PYTHON -u -m jupyter nbconvert --to notebook --execute "$NOTEBOOK" --ExecutePreprocessor.timeout=0 --output "$EXECUTED_NOTEBOOK" 2>&1 | tee -a "$LOGFILE"

EXIT_CODE=${PIPESTATUS[0]:-0}

echo "Finished notebook execution: $(date) (exit code: $EXIT_CODE)" | tee -a "$LOGFILE"
exit $EXIT_CODE
