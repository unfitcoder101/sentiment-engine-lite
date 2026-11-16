import os

import pytest


def pytest_collection_modifyitems(config, items):
    """Tests marked `models` need real local weights; skipped unless SE_RUN_MODELS=1."""
    if os.environ.get("SE_RUN_MODELS") == "1":
        return
    skip = pytest.mark.skip(reason="needs real local model weights (set SE_RUN_MODELS=1 and SE_MODEL_DIR)")
    for item in items:
        if "models" in item.keywords:
            item.add_marker(skip)
