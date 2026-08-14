"""
Shared MLflow / DagsHub tracking setup.

Both `train.py` and `registry.py` need to talk to the *same* MLflow
tracking + model registry server (hosted on DagsHub). Centralizing the
`dagshub.init(...)` call here guarantees both modules end up pointed at the
same place, and makes sure we only initialize the connection once per
process no matter how many times it's imported/called.
"""
import sys
from pathlib import Path
from warnings import filterwarnings

PROJECT_ROOT = str(Path(__file__).resolve().parent.parent.parent)
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

from utils.logger import logging

filterwarnings("ignore")

DAGSHUB_REPO_OWNER = "aniqramzan5758"
DAGSHUB_REPO_NAME = "RevuAI"

_initialized = False


def init_mlflow_tracking() -> None:
    """
    Point MLflow at the DagsHub-hosted tracking + model registry server.

    Safe to call multiple times (and from multiple modules) -- only the
    first call actually does anything; subsequent calls are no-ops. If the
    DagsHub connection can't be established (e.g. no network / no auth),
    we log a warning and let MLflow fall back to its default local
    tracking store rather than crashing the pipeline.
    """
    global _initialized
    if _initialized:
        return

    try:
        import dagshub
        dagshub.init(
            repo_owner=DAGSHUB_REPO_OWNER,
            repo_name=DAGSHUB_REPO_NAME,
            mlflow=True,
        )
        logging.info(
            f"MLflow tracking initialized against DagsHub repo "
            f"'{DAGSHUB_REPO_OWNER}/{DAGSHUB_REPO_NAME}'."
        )
    except Exception as e:
        logging.warning(
            f"DAGsHub initialization skipped: {e}. "
            "Falling back to MLflow's default (local) tracking store."
        )
    finally:
        _initialized = True