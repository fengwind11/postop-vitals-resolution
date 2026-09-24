#!/usr/bin/env sh
set -eu
cd "$(dirname "$0")"
export PYTHONDONTWRITEBYTECODE=1
if [ "${1:-}" = "stage" ]; then
  [ "$#" -eq 2 ] || { echo "Usage: ./run_all.sh stage PRIVATE_WORK_DIRECTORY"; exit 2; }
  exec python prepare_private_workspace.py --work-dir "$2"
fi
[ "$#" -eq 0 ] || { echo "Use no arguments for synthetic test, or stage PRIVATE_WORK_DIRECTORY"; exit 2; }
exec python smoke_test/smoke_test.py
