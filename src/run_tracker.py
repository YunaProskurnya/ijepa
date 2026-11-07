from collections.abc import Mapping
from importlib import import_module
import json
import os


def _flatten_parameters(config, prefix=""):
    result = {}
    for name, value in config.items():
        key = f"{prefix}.{name}" if prefix else str(name)
        if isinstance(value, Mapping) and value:
            result.update(_flatten_parameters(value, key))
        elif isinstance(value, (str, int, float, bool)):
            result[key] = value
        elif value is None:
            result[key] = "null"
        else:
            result[key] = json.dumps(value, default=str)
    return result


class NullRunTracker:
    def log_parameters(self, config):
        return None

    def log_metrics(self, metrics, *, step, epoch):
        return None

    def log_model(self, model_state, model_name="rejepa"):
        return None

    def end(self):
        return None


class CometRunTracker:
    def __init__(self, project, workspace):
        api_key = os.environ.get("COMET_API_KEY", "").strip()
        if not api_key:
            raise RuntimeError("COMET_API_KEY is required; source .env before running")

        comet = import_module("comet_ml")
        pytorch = import_module("comet_ml.integration.pytorch")
        self._experiment = comet.start(api_key=api_key, project_name=project, workspace=workspace)
        self._log_model = pytorch.log_model

    def log_parameters(self, config):
        self._experiment.log_parameters(config)

    def log_metrics(self, metrics, *, step, epoch):
        values = {key: value for key, value in metrics.items() if value is not None}
        if values:
            self._experiment.log_metrics(values, step=step, epoch=epoch)

    def log_model(self, model_state, model_name="rejepa"):
        self._log_model(self._experiment, model_state, model_name=model_name)

    def end(self):
        self._experiment.end()


class ClearMLRunTracker:
    def __init__(self, project, task_name):
        access_key = os.environ.get("CLEARML_API_ACCESS_KEY", "").strip()
        secret_key = os.environ.get("CLEARML_API_SECRET_KEY", "").strip()
        if not access_key or not secret_key:
            raise RuntimeError(
                "ClearML logging requires CLEARML_API_ACCESS_KEY and CLEARML_API_SECRET_KEY."
            )

        clearml = import_module("clearml")
        self._task = clearml.Task.init(
            project_name=project,
            task_name=task_name,
            reuse_last_task_id=False,
            auto_connect_arg_parser=False,
            auto_connect_frameworks=False,
            auto_resource_monitoring=False,
            auto_connect_streams=False,
        )
        self._logger = self._task.get_logger()

    def log_parameters(self, config):
        self._task.connect(_flatten_parameters(config), name="config", ignore_remote_overrides=True)

    def log_metrics(self, metrics, *, step, epoch):
        for name, value in metrics.items():
            if value is None:
                continue
            self._logger.report_scalar(title="train", series=str(name), value=float(value), iteration=int(step))

    def log_model(self, model_state, model_name="rejepa"):
        self._task.upload_artifact(name=model_name, artifact_object=model_state)

    def end(self):
        self._task.close()


def start_run_tracker(backend, *, project, workspace, task_name):
    if backend == "none":
        return NullRunTracker()
    if backend == "comet":
        return CometRunTracker(project, workspace)
    if backend == "clearml":
        return ClearMLRunTracker(project, task_name)
    raise RuntimeError(f"tracker must be comet, clearml, or none; received {backend!r}")
