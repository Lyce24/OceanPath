"""
Shared fixtures and path setup for all OceanPath tests.
"""

import sys
from pathlib import Path

# Ensure src/ is importable (for running pytest from project root)
_src = Path(__file__).parent.parent / "src"
if str(_src) not in sys.path:
    sys.path.insert(0, str(_src))