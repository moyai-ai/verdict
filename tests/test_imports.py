import importlib
from pathlib import Path

import pytest

import verdict

ROOT = Path(verdict.__file__).parent
MODULES = sorted(
    ".".join(("verdict", *path.relative_to(ROOT).with_suffix("").parts))
    for path in ROOT.rglob("*.py")
    if path.name != "__init__.py"
)


@pytest.mark.parametrize("module", MODULES)
def test_import(module):
    importlib.import_module(module)
