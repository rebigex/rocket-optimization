#!/usr/bin/env bash
# Builds the environment from a clean clone, without being asked. Safe to re-run.
#
# This is a thin wrapper: the work lives in bootstrap.py, which is stdlib-only
# and runs on Windows too, so there is one copy of the steps rather than two
# that drift apart.
set -euo pipefail
cd "$(dirname "$0")/.."

# --force, not just --yes: this is the explicit "set my environment up"
# command, so it reinstalls rather than trusting the readiness check,
# which cannot see a dependency that was added since.
python3 bootstrap.py --yes --force
.venv/bin/python -m pytest tests/ -q
