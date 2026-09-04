#!/usr/bin/env bash
# Builds the environment from a clean clone. Safe to re-run.
set -euo pipefail
cd "$(dirname "$0")/.."

OPENMOTOR_COMMIT=0dfb3f1dd4f843499c7f71dc85a3dfde5dd15c6a

python3 -m venv .venv
.venv/bin/pip install --upgrade pip
.venv/bin/pip install -r requirements.txt

# openMotor is GPLv3 and is not redistributed with this project -- it is cloned
# here, pinned, so the two stay separable. See NOTICE.md.
if [ ! -d vendor/openMotor/.git ]; then
  mkdir -p vendor
  git clone https://github.com/reilleya/openMotor.git vendor/openMotor
fi
git -C vendor/openMotor fetch --quiet origin "$OPENMOTOR_COMMIT" || true
git -C vendor/openMotor checkout --quiet "$OPENMOTOR_COMMIT"

# motorlib ships a Cython extension, and its setup.py is too old for pip to
# install editable, so build it in place and put it on the path with a .pth.
( cd vendor/openMotor && ../../.venv/bin/python setup.py build_ext --inplace )
SITE=$(.venv/bin/python -c "import sysconfig; print(sysconfig.get_paths()['purelib'])")
echo "$(pwd)/vendor/openMotor" > "$SITE/openmotor_vendor.pth"

# The app serves Plotly from disk so it works with no network. The JS ships
# inside the plotly pip package; copy it rather than pulling from a CDN.
PLOTLY_JS=$(.venv/bin/python -c "import plotly, pathlib; print(pathlib.Path(plotly.__file__).parent / 'package_data' / 'plotly.min.js')")
mkdir -p app/static/vendor && cp "$PLOTLY_JS" app/static/vendor/plotly.min.js

.venv/bin/python -c "from motorlib.motor import Motor; print('openMotor ready')"
.venv/bin/python -m pytest tests/ -q

echo
echo "Run the app with:  .venv/bin/python app.py"
