# rejepa

A PyTorch reproduction of I-JEPA: our encoder/predictor, Smooth L1, AdamW, and EMA.
Training on a single device (CPU or GPU) is supported.

## Setup Comet/ClearML

On a new machine, create `.env.credentials` in the repository root. `start_setup.sh` will exports them.

Comet: open [Account settings → API Keys](https://www.comet.com/account/apiKeys)
and copy the API key.

ClearML: open [app.clear.ml settings → Workspace → Create new credentials](https://app.clear.ml/settings/workspace-configuration)
and copy the access key and secret key. The three hosts below are the public ClearML app.

```sh
cat > .env.credentials << 'EOF'
export COMET_API_KEY="paste-comet-api-key"
export CLEARML_WEB_HOST="https://app.clear.ml"
export CLEARML_API_HOST="https://api.clear.ml"
export CLEARML_FILES_HOST="https://files.clear.ml"
export CLEARML_API_ACCESS_KEY="paste-clearml-access-key"
export CLEARML_API_SECRET_KEY="paste-clearml-secret-key"
EOF
chmod 600 .env.credentials
```

## Install dependences and smoke test

```sh
python -m venv .env
source start_setup.sh
python -m pip install -r requirements.txt

python scripts/train.py --config configs/smoke.json
```

`source start_setup.sh` loads `.env.credentials` and activates `.env`. 

This check uses 8 synthetic images and a compact model on CPU.
It downloads nothing and saves the configuration, metrics, and a checkpoint.
Each new run needs its own `--output` directory.


## ImageNet100

The commands live in `scripts/setup_imagenet100.sh`. Instead of the block above,
you can run `bash -e ../../scripts/setup_imagenet100.sh` from the `data/imagenet100` directory.
After setup you get `data/imagenet100/train/<class>/` and
`data/imagenet100/val/<class>/`. The training config already uses this path.

The default layout is ImageFolder: `root/class_name/image.jpg`.
Pass the train root for training; I-JEPA does not use class labels.
Evaluation needs separate train and val trees with the same class folder names.
The class list is saved in the run configuration. Data is not downloaded automatically.

Everything in `data/` except the [instructions](data/README.md) is excluded from Git.
If the dataset is already on the server,
point `data` in the config at it, or pass `--data`. You do not need to copy it into the project.

## First GPU run

Run this inside an allocated GPU job or cluster container that has
one H200. The script uses a single device; start it with plain `python`.
Environment check and a 10-epoch run:

```sh
python -c "import torch; assert torch.cuda.is_available(), 'CUDA unavailable'; print(torch.cuda.get_device_name()); print('BF16:', torch.cuda.is_bf16_supported())"
python scripts/train.py --config configs/imagenet100_vit_base.json
```

Base config: ViT-B/16, BF16, batch size 128 with no gradient accumulation,
our collator, and **10 full epochs**, with no step limit.
One epoch is warmup. The learning rate rises from `6.25e-6` to `6.25e-5`,
then decays by cosine to `6.25e-8`. Those three values come from
[I-JEPA, Appendix A.1](https://arxiv.org/pdf/2301.08243)
and are multiplied by `128 / 2048` for the smaller batch. This scaling is our
starting choice for a short run; its quality still needs to be checked.
The goal of the run is to check training stability on ImageNet100 and then evaluate
representation quality with kNN.

A successful run writes `config.json`, `metrics.csv`, and `checkpoint.pt` under
`runs/imagenet100-vit-base-10ep/`. In the metrics, look at `loss`, `grad_norm`,
`steady_images_per_second`, and `cuda_peak_memory_mb`.
For another run, set a new `output`. If you run out of memory, reduce the batch size.

## Experiment configs

`configs/` holds run plans: flat JSON with parameter names in `snake_case`
(`batch_size`, `warmup_epochs`, `max_steps`). Command-line arguments
override the JSON; omitted settings come from the defaults.
Relative paths are resolved from the current working directory.
Run every command here from the repository root.

The main config is `configs/imagenet100_vit_base.json`. If needed, change
`data` to the train path and `output` to a separate results directory.
`configs/smoke.json` is only for a quick CPU check.
In the main config, `max_steps: null` runs all requested epochs.
The `max_steps` limit is the total number of optimizer steps;
schedule length is still set by `epochs`.

A one-off change is easiest on the CLI:

```sh
python scripts/train.py --config configs/imagenet100_vit_base.json --batch-size 32 --accumulation-steps 4 --output runs/imagenet100-vit-base-10ep-small-batch
```

Every image goes through random resized crop to 224×224 and ImageNet normalization.
Masks are regenerated for each batch.

The CUDA launcher disables only the cuDNN backend of SDPA: on an H200 with PyTorch 2.14.1 / cuDNN 9.24, attention produced NaN gradients with finite inputs and context length 64. FlashAttention, memory-efficient, and math SDPA stay available. This does not disable cuDNN for convolutions and does not change `--cudnn-benchmark`.

## Training modes and regularization

`--mode ema` trains the encoder against a frozen target encoder updated by EMA.
`--mode joint` uses a single trainable encoder. For joint, set
`--regularizer sigreg` or `visreg` and, if needed, `--regularizer-weight`.
The same parameters are available in JSON as `mode`, `regularizer`, and `regularizer_weight`.
The 600-epoch configs explicitly set `ema` and `none`. For joint, change both
and set a new `output`.

The default weight `0.02` is taken for SIGReg from the [comparison in the VISReg paper](https://arxiv.org/html/2606.02572).
The VISReg authors used other weights. Those weights cannot be copied onto our
loss as-is: they use `(1−λ)·prediction + λ·regularization`, while we use
`prediction + regularizer_weight·regularization`.

For SIGReg, add `--mode joint --regularizer sigreg` and a separate `--output`
to the VM launch command below. For VISReg, replace `sigreg` with `visreg` and choose the weight explicitly.
The mode and regularizer cannot be changed when resuming a run. The formulas are described
in [src/regularizers/README.md](src/regularizers/README.md).

The default collator is ours: `--collator custom`, implemented
in `src/datasets/imagenet100.py`. The authors' variant is `--collator reference`.
Both return the same batch format, so the encoder, predictor, and training step stay unchanged.
To check the authors' variant:

```sh
python scripts/train.py --config configs/smoke.json --collator reference --output runs/reference-smoke
```

Both variants follow the reference rule: after 20 failed attempts to get a
large enough context, some constraints on overlap with targets are
gradually dropped. With the current settings, the masks of both collators have been
checked to match.

Main parameters:

| Parameter | Purpose |
| --- | --- |
| `--model tiny / vit_base / vit_large` | Default `tiny` for checks. `vit_base` is ViT-B/16: encoder 768×12 blocks, predictor 384×6 blocks. |
| `--device cpu / cuda` | Default CPU. |
| `--batch-size`, `--accumulation-steps` | Effective batch size is their product. The optimizer, EMA, and schedules update after accumulation. |
| `--precision fp32 / bf16` | Default FP32. BF16 reduces memory use on supported hardware. |
| `--compile` | Compiles the encoder and predictor with `Module.compile(dynamic=True, mode="default")` after the checkpoint is loaded. Off by default. |
| `--fused-adamw`, `--cudnn-benchmark` | CUDA only; off by default. Fused AdamW keeps the parameter groups and hyperparameters. cuDNN picks the convolution algorithm for the patch embedding. |
| `--epochs`, `--warmup-epochs` | Defaults 1 and 0. Warmup must be shorter than training. |
| `--lr`, `--weight-decay`, `--ema-start` | Peak LR, initial weight decay, and EMA. Other schedule parameters are in `--help`. |
| `--num-workers`, `--seed` | Data loading and reproducibility. Defaults 0 and 0. |

If you run out of memory, reduce the batch size and, if needed, compensate with
gradient accumulation. Masks are consistent inside each microbatch, so
accumulation is not fully equivalent to one large batch.

In the remote configs `imagenet100_vit_base_400ep_sk_container_sigreg.json`
and `imagenet100_vit_base_400ep_sk_container_visreg.json`, all three options are on.
To enable them from the CLI, add `--compile --fused-adamw --cudnn-benchmark`:

```sh
python scripts/train.py --config configs/imagenet100_vit_base_400ep_sk_container_visreg.json \
  --compile --fused-adamw --cudnn-benchmark
```

To turn them off, including on `--resume`:

```sh
python scripts/train.py --config configs/imagenet100_vit_base_400ep_sk_container_visreg.json \
  --no-compile --no-fused-adamw --no-cudnn-benchmark
```

These are execution settings: they can change on resume while the training
recipe stays the same. When AdamW state is loaded, the requested
`fused`/`foreach` settings are applied; the other compatibility checks stay strict.
The options are stored in the run metadata. BF16, batch size, masks, schedules,
loss, and the per-patch regularization loop are kept. Small rounding
differences are acceptable. cuDNN may pick different algorithms between runs.
Compilation adds startup time. Speed on H200, the CUDA compiler,
and downstream accuracy have not been checked yet. The APIs were checked against PyTorch 2.14.
Local checks use `menv` with PyTorch 2.4.1; the environments are left unchanged.
In this macOS environment the default compiler raises a C++ type error on
`div_floor_integer` with dynamic shapes. The local numerical check uses
`aot_eager`. CUDA checks of the default compiler and fused AdamW are skipped without a GPU.
API docs: [Module.compile](https://docs.pytorch.org/docs/2.14/generated/torch.nn.Module.html#torch.nn.Module.compile),
[AdamW](https://docs.pytorch.org/docs/2.14/generated/torch.optim.AdamW.html),
[cuDNN benchmark](https://docs.pytorch.org/docs/2.14/notes/randomness.html#cuda-convolution-benchmarking).

## Results and resume

The `--output` directory stores:

- `config.json` — the full effective run configuration after JSON and CLI are merged;
- `metrics.csv` — loss, gradients, schedules, speed, and GPU memory;
- `checkpoint.pt` — models, optimizer, schedules, random generators, and progress.
- `checkpoint_0.pt`, `checkpoint_1.pt`, … — a separate checkpoint after each fully finished epoch, numbered from zero; saved by default.

`config.json` and `checkpoint["config"]` store the same parameters:
the final arguments, models with defaults filled in, masks, preprocessing,
objective, optimizer, schedules, library versions, and the class list.
The paths `data`, `output`, `resume`, and `config` are written as absolute paths when set.
The `config` field points at the original experiment file.
This snapshot contains computed values and bookkeeping, so
`runs/.../config.json` is not an input file for `--config`.
For new runs, edit the files in `configs/`.

The checkpoint is updated every `--checkpoint-every` optimizer steps (default
100), at the end of an epoch, and on a clean stop. To resume, repeat the original
command with the same `--output` and add `--resume path/to/checkpoint.pt`.
To resume from a specific epoch, use `--resume path/to/checkpoint_N.pt` (N is the epoch index from zero).

There is no separate flag for saving every epoch. Example launch:

```sh
python scripts/train.py --config configs/imagenet100_vit_base.json --output runs/my-run
```

Resume from the first-epoch checkpoint with the same settings and output directory:

```sh
python scripts/train.py --config configs/imagenet100_vit_base.json \
  --output runs/my-run --resume runs/my-run/checkpoint_0.pt
```

`--checkpoint-every 100` writes extra saves to `checkpoint.pt`
every 100 optimizer steps. Per-epoch checkpoints are saved automatically
and take extra disk space.

Data and training parameters, including the collator, total epochs, batch size, and accumulation,
must match. If the step limit is reached, raise `--max-steps`
or set `"max_steps": null` in the original JSON.
If you change the total number of epochs after a check, start a new run with a separate
`output`: an old checkpoint cannot be resumed with a different schedule.
With several DataLoader workers, an exact replay of future crops and masks after
a restart is not guaranteed, because of prefetching.

## Representation evaluation

### During training

When `representation_data` is set, the first epoch, every 25th epoch, and the last finished epoch
are evaluated on one sample. EMA uses the target encoder; joint uses the shared
encoder. Both 600-epoch configs already set `"representation_data": "val"`.
On the VM, start from the directory that contains `train/` and `val/`:

```sh
cd /home/container/shared/rejepa
python /home/container/rejepa/scripts/train.py \
  --config /home/container/rejepa/configs/imagenet100_vit_base_600ep_sk_container.json
```

Outputs: `representation_analysis/representations.csv`, `spectrum_N.pt` with the
spectrum, and `settings.json` with the paths of the selected images. Sample size is
`--representation-samples` (default 1024). On resume, rows are not
duplicated.

### Analysis of saved epochs

Existing `checkpoint_N.pt` files can be analyzed without the training process or Comet.
Epoch comparison uses a fixed resize and center crop; training
uses a random crop. ImageNet normalization is shared. Patch tokens are averaged
over the image. From the covariance we compute the spectrum, its entropy,
`effective_rank = exp(entropy)`, and the mean variance. The covariance is not saved.

```sh
cd /home/container/shared/rejepa
python /home/container/rejepa/scripts/analyze_checkpoints.py \
  --checkpoints runs/imagenet100-vit-base-600ep \
  --data val \
  --device cuda \
  --bf16 \
  --every 25
```

Without `--every`, every checkpoint is analyzed. `--every 25` selects the first,
the last, and every 25th epoch. A second run overwrites the CSV and the selected
spectra in `checkpoint_analysis/` and leaves the online log untouched. For a separate
result, pass `--output`. A low rank and variance can
indicate collapse, but they do not measure class separation.
`checkpoint_analysis/representations.csv` holds the rank of patch-averaged
features, and `checkpoint_analysis/patch_ranks.csv` holds the effective rank for each
patch position on the same fixed image sample. `patch_index` follows
encoder token order, starting at zero. Comparing these files shows whether
the diversity of individual patch features survives averaging.

### kNN

kNN checks how well classes separate in the features of the frozen
**target encoder** for EMA, or the shared **encoder** for joint. Its patch tokens are averaged into one image vector and
normalized. For each val image, the nearest train images by cosine
similarity are found. Their labels vote for the class, with a larger weight for
closer neighbors. No separate classifier is trained.

```sh
python scripts/evaluate_knn.py --checkpoint runs/imagenet100-vit-base-10ep/checkpoint.pt --train data/imagenet100/train --val data/imagenet100/val --device cuda --bf16 --compare-random --output runs/imagenet100-vit-base-10ep/knn.json
```

By default, evaluation runs on CPU with 20 neighbors, 20 reference images from train,
and 5 val queries per class. Change the counts with `--k`, `--reference-per-class`,
and `--queries-per-class`. Each class must have enough images. For GPU,
add `--device cuda`, and `--bf16` when BF16 is supported.

`--compare-random` repeats the evaluation for the same architecture with random initialization
on the same images. This checks the gain from training. A random encoder does not
have to match chance accuracy: it can still preserve image
similarity. Labels are used only at evaluation time, not while training I-JEPA.

The JSON contains top-1/top-5 accuracy, the top-1 gain in percentage points, feature
spread, effective rank, and image lists. When comparing experiments,
keep the data and evaluation parameters the same. kNN does not replace the linear probe
from the paper.

## Final linear probe: EMA, SIGReg, and VISReg

In [configs/linear_probe_cached.json](configs/linear_probe_cached.json), change
`checkpoint`, `train`, `val`, `cache`, and `output` to the paths of your run.
On a server with the dataset at the project root, set `"train": "train"` and `"val": "val"`.
Each checkpoint needs its own cache directory.

```sh
# Once: features of 50 random crop/flip views of each train image,
# plus fixed train/val features for the final metrics.
python scripts/prepare_linear_cache.py --config configs/linear_probe_cached.json

# Linear heads only, with no extra encoder passes.
python scripts/evaluate_linear_cached.py --config configs/linear_probe_cached.json
```

The script evaluates a frozen encoder: the EMA target for I-JEPA, or the shared encoder
for SIGReg/VISReg. The cache holds features of one checkpoint and lets you repeat
evaluation without new encoder passes. Another checkpoint needs its own cache.
The final evaluation sweeps 24 head variants (12 for a small encoder with fewer
than four blocks), trains them with LARS for 50 epochs, and picks the best by val top-1.
Comet is off for evaluation by default. Feature preparation takes time.

| File | Contents |
| --- | --- |
| `linear_cached.best.md` | Readable best result with parameters and train/val accuracy and loss |
| `linear_cached.best.json` | Best result for programmatic reading |
| `linear_cached.trials.csv` | All variants as a table |
| `linear_cached.json` | Full protocol and metrics |

## Lightweight linear-probe curve over saved epochs

[configs/linear_probe_curve.json](configs/linear_probe_curve.json) sets the
`checkpoints` directory, the `train` / `val` paths, and a separate `output` directory for results.
One launch evaluates one trained model (EMA, SIGReg, or VISReg):

```sh
python scripts/evaluate_linear_curve.py --config configs/linear_probe_curve.json
```

By default, **every saved epoch** is checked. `every: 10` keeps the first,
the last, and every tenth finished epoch. Supported names are `checkpoint_0.pt`
(the first epoch in the training loop from main) and `checkpoint_epoch_0010.pt`
(the tenth epoch in experiment monitoring). The overwritten `checkpoint.pt` is not included.

For speed, up to 100 train images per class and the full val set are fixed:
for ImageNet100 that is about 10 thousand train images and 5 thousand val images. The selected file list
is saved in `selection.json` and is the same for every checkpoint and method when the
data, classes, and seed match. `train_per_class: 0` uses the full train set;
`val_per_class: 0` uses the full val set. For each checkpoint the encoder processes
the selected train/val images once. Features stay in RAM only for the current point.
The head is reinitialized each time with the same seed: mean of the last layer,
Linear with bias, Adam, LR 0.001, 30 epochs, batch 256. There is no sweep and no transfer of head weights.

This is a **lightweight diagnostic probe**. Its accuracy is not the final LARS evaluation
from the section above. The X axis is the encoder pretraining epoch, not the head training epoch.
Use the same settings and data when comparing methods.

`output` stores `linear_curve.csv` (train/val accuracy, loss, epoch, step,
and time of each point), `settings.json`, `selection.json`, and the plots
`linear_curve.png` / `linear_curve.svg`. The CSV is updated after each point.
Running the command again skips finished checkpoints and continues;
if the data, settings, or already evaluated files change, it requires a new directory.
You can run it again during pretraining: newly
saved checkpoints will be added. The script exits after the files it found;
it does not wait for the next epoch. On a shared GPU, evaluation uses
compute and can slow pretraining down.
You can change `every` to fill in a sparse curve over time. While it runs,
it prints the time per point and an estimate of the time left. Even a lightweight probe over hundreds of
checkpoints takes time: start with a few epochs at `every: 50`.
Comet and an API key are not required. For the plots, install the updated `requirements.txt`.

Compare finished curves for a presentation, without a GPU:

```sh
python scripts/plot_linear_curves.py \
  --csv runs/ema/linear_curve/linear_curve.csv \
        runs/sigreg/linear_curve/linear_curve.csv \
        runs/visreg/linear_curve/linear_curve.csv \
  --labels EMA SIGReg VISReg --output runs/linear_probe_comparison
```

## Where to read and change the code

`src/` holds the method, `scripts/` the training and evaluation entry points,
`configs/` the experiment plans, `example/` the tutorial examples,
`data/` the local datasets, and `runs/` the local results.

- `src/models/` — encoder, predictor, and target-encoder/EMA helpers.
- `src/train_step.py` — one step: targets → predictions → loss → backward → optimizer → EMA.
- `src/optimizer.py` and `src/losses.py` — AdamW parameter groups and the loss.
- `scripts/train.py` — arguments, DataLoader, loop, schedules, and saving.
- `scripts/evaluate_knn.py` — evaluation of a saved model.
- `scripts/prepare_linear_cache.py`, `scripts/evaluate_linear_cached.py` — feature preparation and the final linear probe.
- `src/linear_probe_final.py` — the shared LARS protocol; `src/linear_probe_cache.py` — the on-disk cache.
- `scripts/setup_imagenet100.sh` — the team's commands for downloading and laying out ImageNet100.
- `src/config.py` — JSON loading and run-setting checks.
- `src/reference/` — the authors' collator and LR/weight-decay schedules.
- `src/datasets/imagenet100.py` — our collator and the shared DataLoader setup.
  `make_imagenet100` reads `root/train` or `root/val` and returns the dataset, loader, and sampler.
  If you use it on its own during training, call `sampler.set_epoch(epoch)` before each epoch.

The `reference` files are taken unchanged from
[facebookresearch/ijepa, commit 52c1ae95](https://github.com/facebookresearch/ijepa/tree/52c1ae95d05f743e000e8f10a1f3a79b10cff048).
The source and the CC BY-NC 4.0 license are next to them. Our implementation was ported
from the `data_encoder` branch (commit `930120e`) with adapted interfaces.
Both collators keep the batch contract:
`((images, labels), [context_indices], [target_indices_1, ...])`.
Indices refer to the patch grid after crop/resize. Each set has shape `[B, K]`.

Full parameter list and checks:

```sh
python scripts/train.py --help
python scripts/evaluate_knn.py --help
python -m unittest discover -s tests
```
