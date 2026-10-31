# Gaussian noise and I-JEPA prediction

## Question and result so far

Can independent Gaussian noise on normalized full-image target representations, followed by an MSE prediction objective, prevent collapse in joint I-JEPA training without SIGReg or VISReg?

The answer is **no as a general guarantee**. Independent target noise adds a fixed noise floor to the expected MSE; a constant encoder and predictor can still attain that floor. Noise may alter optimization enough to improve representations empirically. That narrower claim needs training runs; none have been completed for this branch.

The [I-JEPA paper](https://arxiv.org/abs/2301.08243) defines masked context to target prediction. The [noise-injected Deep InfoMax paper](https://proceedings.iclr.cc/paper_files/paper/2025/file/c8ff6807d1f362bb22b4f0be7b66e9ca-Paper-Conference.pdf) adds independent noise to normalized encoder outputs **while maximizing a mutual information objective**. Its Gaussian theorem assumes zero mean, unit population variance in every feature, independent isotropic Gaussian noise, and conditions on paired views; equality also requires weak invariance and a Gaussian representation distribution. [AWGN capacity](https://ocw.mit.edu/courses/6-441-information-theory-spring-2016/24fb44b6a101e847c32ffe38bbeeea6e_MIT6_441S16_chapter_17.pdf) maximizes mutual information between a channel input and its noisy output under an average power constraint. Neither result says that minimizing a learned target's prediction error maximizes that information. Standardizing one or both branches does not turn MSE prediction into the paper's InfoMax objective.

## Objective and counterexample

Let `C = f_theta(image, context_mask)`, `T = select(LN(f_theta(image)), target_mask)`, and let `N ~ Normal(0, sigma^2 I)` be independent of the image, mask, and parameters. Noise is added to the normalized full-image output before target patches are selected. The experiment minimizes

`E[||g_phi(C, target_positions) - (T + N)||^2]`.

Because the noise is independent and zero mean, the expected per-element MSE equals the clean-target MSE plus `sigma^2`. A constant encoder makes `T` constant; a constant predictor then reaches the noise floor. The current tokenwise LayerNorm does not constrain variation across images. BatchNorm with `affine=False` is not implemented, and even a constant input to BatchNorm produces zero output, so it would not remove this counterexample.

An MSE-based lower bound on predictive mutual information includes the entropy of the noisy target. That entropy changes with the jointly trained encoder; at collapse it is only the entropy of the added noise. Reducing MSE therefore does not by itself imply increasing mutual information or approaching an isotropic Gaussian representation.

## Training experiment

[imagenet100_vit_base_400ep_sk_container_authoreg.json](../configs/imagenet100_vit_base_400ep_sk_container_authoreg.json) matches the existing Skoltech container configs: full ImageNet100, 400 epochs, ViT Base, and output under `/home/container/shared/rejepa/runs/`. It uses joint encoder training, MSE, no regularizer, and target noise `sigma=0.1`. Noise is sampled independently for each full-image target token after LayerNorm and before target patch selection. Context embeddings stay clean. Evaluation uses clean encoder features. The original smooth L1 and zero noise settings remain the defaults for existing runs.

The launcher requires `COMET_API_KEY`; it saves `config.json`, `metrics.csv`, checkpoints, and representation spectra in the output directory. Check GPU availability, run training in tmux, and save terminal output:

```sh
cd /home/container/shared/rejepa
mkdir -p logs
nvidia-smi
tmux new -s authoreg
python /home/container/rejepa/scripts/train.py \
  --config /home/container/rejepa/configs/imagenet100_vit_base_400ep_sk_container_authoreg.json \
  2>&1 | tee logs/authoreg-400ep.log
```

The working directory must contain `train/` and `val/`. For matched full length ablations, override `--noise-std` with `0` or `0.5` and give each run a distinct `--output` and log name. Use `--regularizer sigreg --noise-std 0` for a positive control. Repeat with matched seeds if resources permit. Do not compare raw loss across SIGReg and unregularized runs as a collapse measure.

Compare clean feature mean variance and effective rank from `representation_analysis/representations.csv` at matched epochs, then assess downstream utility with the repository's kNN or linear probe evaluation on held-out data. Loss alone can fall while the representation collapses. Diagnostics are recorded at the first, every 25th, and final completed epoch. [VISReg](https://arxiv.org/abs/2606.02572) and [VICReg](https://arxiv.org/abs/2105.04906) motivate the positive regularization comparison.

## Current limits

No ImageNet100 training was run. The default Python lacks PyTorch; an alternate local environment has PyTorch but lacks torchvision and Comet. This workspace also has no prepared dataset, usable GPU, or tmux, and the configured remote server is unavailable. The analytical counterexample settles the universal anti-collapse claim, while the empirical effect of noise on this implementation remains open.
