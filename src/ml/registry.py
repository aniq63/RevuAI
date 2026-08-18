import sys
from pathlib import Path
from typing import Optional

import mlflow
from mlflow import MlflowClient

PROJECT_ROOT = str(Path(__file__).resolve().parent.parent.parent)
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

from utils.logger import logging
from utils.exception import MyException


class ModelRegistry:
    """
    Registers a model only if it outperforms the current Champion model
    and passes the configured minimum quality gate.
    """

    def __init__(
        self,
        run_id: str,
        test_f1: float,
        model_name: str = "linear_svc_classifier",
        artifact_path: str = "linear_svc_model",
        champion_alias: str = "Champion",
        always_promote: bool = False,
        min_test_f1: Optional[float] = None,
    ):

        self.client = MlflowClient()

        self.run_id = run_id
        self.test_f1 = test_f1

        self.model_name = model_name
        self.artifact_path = artifact_path
        self.champion_alias = champion_alias
        self.always_promote = always_promote

        # Minimum test F1 a model must reach to be registered/promoted.
        # When set, a model below this gate is never registered and the
        # current Champion (if any) is kept.
        self.min_test_f1 = min_test_f1

    # -------------------------------------------------------
    # Helpers
    # -------------------------------------------------------

    def _model_uri(self):
        return f"runs:/{self.run_id}/{self.artifact_path}"

    def _registered_model_exists(self):

        try:
            self.client.get_registered_model(self.model_name)
            return True

        except Exception:
            return False

    def _create_registered_model(self):

        logging.info(
            f"Creating Registered Model : {self.model_name}"
        )

        self.client.create_registered_model(self.model_name)

    def _register_new_version(self):

        logging.info("Registering a new model version...")

        mv = self.client.create_model_version(
            name=self.model_name,
            source=self._model_uri(),
            run_id=self.run_id,
        )

        return mv

    def _set_champion(self, version):

        self.client.set_registered_model_alias(
            self.model_name,
            self.champion_alias,
            version,
        )

    def _get_champion(self):

        return self.client.get_model_version_by_alias(
            self.model_name,
            self.champion_alias,
        )

    def _champion_f1(self):

        champion = self._get_champion()

        run = self.client.get_run(champion.run_id)

        return float(run.data.metrics["test_f1"])

    # -------------------------------------------------------
    # Main Function
    # -------------------------------------------------------

    def register(self):

        try:

            # ---------------------------------------------
            # Minimum quality gate
            # ---------------------------------------------

            below_gate = (
                self.min_test_f1 is not None
                and self.test_f1 < self.min_test_f1
            )

            if below_gate:
                logging.warning(
                    f"New model test F1 ({self.test_f1:.4f}) is below the "
                    f"minimum quality threshold ({self.min_test_f1:.4f})."
                )

            # ---------------------------------------------
            # First model ever
            # ---------------------------------------------
            if not self._registered_model_exists():

                logging.info(
                    "No registered model found."
                )

                if below_gate:
                    logging.error(
                        "Cannot register a model below the quality gate "
                        "when no Champion exists yet."
                    )
                    raise MyException(
                        "New model test F1 is below the minimum quality "
                        "threshold and no Champion exists.",
                        sys,
                    )

                self._create_registered_model()

                version = self._register_new_version()

                self._set_champion(version.version)

                logging.info(
                    "Registered first model successfully."
                )

                return version

            # ---------------------------------------------
            # Existing Champion
            # ---------------------------------------------

            champion = self._get_champion()

            champion_f1 = self._champion_f1()

            logging.info(
                f"Champion Test F1 : {champion_f1:.5f}"
            )

            logging.info(
                f"Current Test F1 : {self.test_f1:.5f}"
            )

            # Never replace the Champion with a below-threshold model,
            # even when always_promote is enabled.
            if below_gate:

                logging.warning(
                    "Keeping the current Champion (model is below the "
                    "quality threshold)."
                )

                return champion

            # ---------------------------------------------
            # Compare
            # ---------------------------------------------

            if self.always_promote or self.test_f1 > champion_f1:

                logging.info(
                    "New model is better."
                )

                version = self._register_new_version()

                self._set_champion(version.version)

                logging.info(
                    f"Champion updated to version {version.version}"
                )

                return version

            logging.info(
                "Champion model is still better."
            )

            return champion

        except Exception as e:

            raise MyException(e, sys)