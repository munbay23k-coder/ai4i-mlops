import pytest

from ai4i_mlops.config import MODEL_PATH, METRICS_PATH
from ai4i_mlops.train import main


@pytest.fixture(scope="session")
def trained_model():
    if not MODEL_PATH.exists() or not METRICS_PATH.exists():
        main()

    return MODEL_PATH