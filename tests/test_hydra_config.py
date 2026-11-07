from contextlib import contextmanager
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

from hydra import compose, initialize_config_dir
from hydra.core.global_hydra import GlobalHydra
from hydra.core.hydra_config import HydraConfig
from omegaconf import OmegaConf


ROOT = Path(__file__).resolve().parents[1]
RUNS = "/home/container/shared/rejepa/runs"
EXPERIMENTS = ROOT / "configs_hydra" / "experiment"


@contextmanager
def compose_experiment(name):
    GlobalHydra.instance().clear()
    with initialize_config_dir(config_dir=str(ROOT / "configs_hydra"), version_base="1.3"):
        cfg = compose(
            config_name="config",
            overrides=[f"experiment={name}"],
            return_hydra_config=True,
        )
        HydraConfig.instance().set_config(cfg)
        yield cfg


class TestHydraConfigs(unittest.TestCase):
    def tearDown(self):
        GlobalHydra.instance().clear()

    def experiment_names(self):
        return sorted(path.stem for path in EXPERIMENTS.glob("*.yaml"))

    def test_experiments_compose_from_their_own_system_and_training(self):
        names = self.experiment_names()
        self.assertIn("smoke", names)
        self.assertGreaterEqual(len(names), 2)
        for name in names:
            with self.subTest(experiment=name):
                with compose_experiment(name) as cfg:
                    system = OmegaConf.to_container(cfg.system, resolve=True)
                    training = OmegaConf.to_container(cfg.training, resolve=True)
                self.assertEqual(system["output"], f"{RUNS}/{name}")
                self.assertIn(training["model"], ("tiny", "vit_base", "vit_large"))
                self.assertGreater(training["epochs"], 0)
                self.assertGreater(training["batch_size"], 0)
                if name == "smoke":
                    self.assertEqual(system["tracker"], "none")
                    self.assertEqual(system["device"], "cpu")
                    self.assertTrue(system["smoke_test"])
                    self.assertIsNone(system["data"])
                else:
                    self.assertEqual(system["tracker"], "comet")
                    self.assertEqual(system["device"], "cuda")
                    self.assertFalse(system.get("smoke_test", False))

    def test_hydra_smoke_step_uses_none_tracker(self):
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "run"
            env = os.environ.copy()
            env.pop("COMET_API_KEY", None)
            env.pop("CLEARML_API_ACCESS_KEY", None)
            env.pop("CLEARML_API_SECRET_KEY", None)
            result = subprocess.run(
                [sys.executable, str(ROOT / "scripts/train_hydra.py"),
                 "experiment=smoke", "training.max_steps=1", f"system.output={output}"],
                cwd=ROOT, capture_output=True, text=True, timeout=120, env=env,
            )
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            config = json.loads((output / "config.json").read_text(encoding="utf-8"))
            self.assertEqual(config["tracker"], "none")
            self.assertEqual(config["model"], "tiny")
            self.assertTrue(config["smoke_test"])
            self.assertTrue((output / "checkpoint.pt").exists())
            self.assertTrue((output / "metrics.csv").exists())
            self.assertFalse((output / "comet.json").exists())


if __name__ == "__main__":
    unittest.main()
