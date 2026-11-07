import os
import unittest
from unittest.mock import Mock, patch

from src.run_tracker import start_run_tracker


class TestRunTracker(unittest.TestCase):
    def test_none_does_not_import_sdks(self):
        with patch("src.run_tracker.import_module") as import_module:
            tracker = start_run_tracker("none", project="rejepa", workspace="superustam", task_name="smoke")
            tracker.log_parameters({"batch_size": 1})
            tracker.log_metrics({"loss": None}, step=1, epoch=1)
            tracker.log_model({"encoder": {}}, model_name="rejepa")
            tracker.end()
        import_module.assert_not_called()

    def test_missing_credentials_fail_before_sdk_start(self):
        with patch("src.run_tracker.import_module") as import_module:
            with patch.dict(os.environ, {"COMET_API_KEY": ""}, clear=False):
                with self.assertRaisesRegex(RuntimeError, "COMET_API_KEY"):
                    start_run_tracker("comet", project="rejepa", workspace="superustam", task_name="run")
            with patch.dict(os.environ, {"CLEARML_API_ACCESS_KEY": "", "CLEARML_API_SECRET_KEY": ""}, clear=False):
                with self.assertRaisesRegex(RuntimeError, "CLEARML_API_ACCESS_KEY"):
                    start_run_tracker("clearml", project="rejepa", workspace="superustam", task_name="run")
        import_module.assert_not_called()

    def test_unknown_backend_is_rejected(self):
        with self.assertRaisesRegex(RuntimeError, "tracker must be"):
            start_run_tracker("wandb", project="rejepa", workspace="superustam", task_name="run")

    def test_comet_logs_parameters_metrics_and_model(self):
        comet = Mock()
        pytorch = Mock()

        def import_module(name):
            if name == "comet_ml":
                return comet
            if name == "comet_ml.integration.pytorch":
                return pytorch
            raise ImportError(name)

        with patch.dict(os.environ, {"COMET_API_KEY": "test-key"}), patch(
            "src.run_tracker.import_module", side_effect=import_module,
        ):
            tracker = start_run_tracker("comet", project="rejepa", workspace="superustam", task_name="run")
            tracker.log_parameters({"batch_size": 2})
            tracker.log_metrics({"loss": 0.5, "eta_seconds": None}, step=1, epoch=2)
            tracker.log_model({"encoder": "state"}, model_name="rejepa")
            tracker.end()

        comet.start.assert_called_once_with(api_key="test-key", project_name="rejepa", workspace="superustam")
        experiment = comet.start.return_value
        experiment.log_parameters.assert_called_once_with({"batch_size": 2})
        experiment.log_metrics.assert_called_once_with({"loss": 0.5}, step=1, epoch=2)
        pytorch.log_model.assert_called_once_with(experiment, {"encoder": "state"}, model_name="rejepa")
        experiment.end.assert_called_once()

    def test_clearml_logs_parameters_metrics_and_model(self):
        clearml = Mock()
        with patch.dict(os.environ, {"CLEARML_API_ACCESS_KEY": "access", "CLEARML_API_SECRET_KEY": "secret"}), patch(
            "src.run_tracker.import_module", return_value=clearml,
        ):
            tracker = start_run_tracker("clearml", project="rejepa", workspace="superustam", task_name="run-name")
            tracker.log_parameters({"batch_size": 2, "nested": {"lr": 0.1}, "empty": {}})
            tracker.log_metrics({"loss": 0.25, "eta_seconds": None}, step=3, epoch=1)
            state = {"encoder": {"weight": 1}}
            tracker.log_model(state, model_name="rejepa")
            tracker.end()

        clearml.Task.init.assert_called_once_with(
            project_name="rejepa",
            task_name="run-name",
            reuse_last_task_id=False,
            auto_connect_arg_parser=False,
            auto_connect_frameworks=False,
            auto_resource_monitoring=False,
            auto_connect_streams=False,
        )
        task = clearml.Task.init.return_value
        task.connect.assert_called_once_with(
            {"batch_size": 2, "nested.lr": 0.1, "empty": "{}"},
            name="config",
            ignore_remote_overrides=True,
        )
        task.get_logger.return_value.report_scalar.assert_called_once_with(
            title="train", series="loss", value=0.25, iteration=3,
        )
        task.upload_artifact.assert_called_once_with(name="rejepa", artifact_object=state)
        task.close.assert_called_once()


if __name__ == "__main__":
    unittest.main()
