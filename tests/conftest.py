"""
Shared fixtures.

Tests mutate the `_config.C` singleton via `load_config`. This fixture
snapshots and restores C around each test so they can't pollute each other.
"""

import pytest

from swarmflow._config import C


@pytest.fixture(autouse=True)
def reset_C():
    saved = dict(C.__dict__)
    yield
    C.__dict__.clear()
    C.__dict__.update(saved)
