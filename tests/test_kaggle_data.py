import json
from pathlib import Path
import tempfile
import unittest

from scripts.prepare_kaggle_imagenet100 import main


class TestKaggleData(unittest.TestCase):
    def test_sharded_selection_and_manifest(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "imagenet100"
            for split, shards, images in (("train", ("X1", "X2"), 4),
                                          ("val", ("X",), 2)):
                for class_index in range(4):
                    directory = source / f"{split}.{shards[class_index % len(shards)]}" / f"class{class_index}"
                    directory.mkdir(parents=True)
                    for image_index in range(images):
                        (directory / f"{image_index}.jpg").write_bytes(b"image")
            manifest = root / "selection.json"
            output = root / "first"
            options = ["--input", str(source), "--manifest", str(manifest),
                       "--classes", "2", "--train-per-class", "3", "--seed", "3"]
            main([*options, "--output", str(output)])
            selected = json.loads(manifest.read_text(encoding="utf-8"))
            self.assertEqual(len(selected["classes"]), 2)
            for name in selected["classes"]:
                self.assertEqual(len(list((output / "train" / name).iterdir())), 3)
                self.assertEqual(len(list((output / "val" / name).iterdir())), 2)
                self.assertTrue(all(path.is_symlink() for path in (output / "train" / name).iterdir()))
            main([*options, "--output", str(root / "second")])
            self.assertEqual(json.loads(manifest.read_text(encoding="utf-8")), selected)


if __name__ == "__main__":
    unittest.main()
