"""Makes the test package and the source tree importable.

There is no motor fixture here: the sample motor is built in code by
:mod:`tests.sample_motor`, so the repository holds no .ric of its own.
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))
