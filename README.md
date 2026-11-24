# semg2pose

Compact pose representations for generalizable sEMG-to-pose decoding, built on top of
Meta's [emg2pose](https://github.com/facebookresearch/emg2pose) benchmark.

Research question: does compressing hand pose into a low-dimensional representation
(PCA or autoencoder latent) before decoding it from wrist sEMG improve generalization to
unseen users and unseen movement stages, and what is the smallest representation size that
keeps decoding accuracy within a pre-specified tolerance?

## Layout

```
semg2pose/
├── emg2pose/          # git submodule → github.com/eljoserass/emg2pose (fork of facebookresearch/emg2pose)
├── src/semg2pose/     # our code (pose latents, bottleneck decoders, evaluation)
├── notebooks/         # exploratory analyses
└── pyproject.toml     # uv project; installs emg2pose + UmeTrack from the submodule (editable)
```

## Setup

Requires [uv](https://docs.astral.sh/uv/). Python 3.10–3.12 (uv picks 3.12 automatically).

```shell
git clone --recurse-submodules https://github.com/eljoserass/semg2pose.git
cd semg2pose

uv sync --extra cpu      # no GPU
# uv sync --extra cu121  # NVIDIA GPU (CUDA 12.1)

# Jupyter kernel for the notebooks
uv run python -m ipykernel install --user --name semg2pose --display-name semg2pose
```

Already cloned without `--recurse-submodules`? Run `git submodule update --init`.

### Data

The notebooks read the mini dataset (~600 MiB) from `~/emg2pose_dataset_mini`:

```shell
cd ~ && curl https://fb-ctrl-oss.s3.amazonaws.com/emg2pose/emg2pose_dataset_mini.tar -o emg2pose_dataset_mini.tar
tar -xvf emg2pose_dataset_mini.tar
```

The full dataset (431 GiB) is described in the
[emg2pose README](https://github.com/facebookresearch/emg2pose#download-the-full-dataset-431-gib).

## Working with the emg2pose submodule

emg2pose is installed in editable mode, so changes inside `emg2pose/` take effect immediately.
Commit them in the submodule (they go to the fork), then commit the new pointer here:

```shell
cd emg2pose && git commit -am "fix: ..." && git push && cd ..
git add emg2pose && git commit -m "bump emg2pose"
```

Pull the latest commit of the fork's tracked branch (`deprecating_conda`):

```shell
git submodule update --remote emg2pose
git add emg2pose && git commit -m "bump emg2pose"
```

### Colab

Every notebook starts with a setup cell that only acts on Colab: it clones this repo with the
submodule and runs `scripts/colab_setup.sh` (pip install of torch 2.3.1 + the three editable
packages). Select *Runtime → Change runtime type → Runtime version 2026.07* (Python 3.12) first,
run the setup cell, restart the session when asked, then run the notebook from the top.

## Notebooks

Shared code (data loading, session table, filters) lives in `src/semg2pose/` and is imported by
the notebooks: `from semg2pose.data import load_session, JOINT_NAMES, ...`.

| Notebook | What it does | |
|---|---|---|
| `motor_units_zoom.ipynb` | Raw EMG, electrode geometry, motor-unit simulation, spectral bands | [Open in Colab](https://colab.research.google.com/github/eljoserass/semg2pose/blob/main/notebooks/motor_units_zoom.ipynb) |
| `emg_angle_correlation.ipynb` | Channel × joint correlations, EMG→movement lag, ridge/CCA decoding | [Open in Colab](https://colab.research.google.com/github/eljoserass/semg2pose/blob/main/notebooks/emg_angle_correlation.ipynb) |
| `joint_space_manifold.ipynb` | Range of motion, PCA eigen-postures, intrinsic dimension, UMAP, PCA vs autoencoder | [Open in Colab](https://colab.research.google.com/github/eljoserass/semg2pose/blob/main/notebooks/joint_space_manifold.ipynb) |
| `pipeline_overview.ipynb` | Shapes and preprocessing end to end, the pretrained vemg2pose layer by layer, metrics, where the bottleneck goes | [Open in Colab](https://colab.research.google.com/github/eljoserass/semg2pose/blob/main/notebooks/pipeline_overview.ipynb) |
| `compression_tolerance.ipynb` | Santello 1998 replication, dimension ladder, per-PC decodability of the pretrained model, post-hoc bottleneck, tolerance rules | [Open in Colab](https://colab.research.google.com/github/eljoserass/semg2pose/blob/main/notebooks/compression_tolerance.ipynb) |
| `spectral_spatial_fingerprints.ipynb` | Band × electrode maps per stage and per finger movement, movement classification, first-layer filters of the network | [Open in Colab](https://colab.research.google.com/github/eljoserass/semg2pose/blob/main/notebooks/spectral_spatial_fingerprints.ipynb) |
| `emg_kinematics_encoding.ipynb` | Position vs velocity vs acceleration (encoding models), lagged decoding (mTRF), multiple coherence, muscle synergies | [Open in Colab](https://colab.research.google.com/github/eljoserass/semg2pose/blob/main/notebooks/emg_kinematics_encoding.ipynb) |
| `neural_synergies.ipynb` | Exploratory: is the low-dimensional pose structure already in the muscle signal? Crosstalk-controlled EMG dimension, reduced-rank (communication subspace) decoding, decodable vs PCA subspaces, muscle synergies across tasks and hands, validation plan | [Open in Colab](https://colab.research.google.com/github/eljoserass/semg2pose/blob/main/notebooks/neural_synergies.ipynb) |
| `cross_user_invariance.ipynb` | Exploratory, 12 users from the full dataset (download with `scripts/partial_dataset/`): pose subspaces are shared across people beyond structured nulls, EMG subspaces are person-specific; leave-one-user-out pose bases | needs local data |
| `intermuscular_coherence.ipynb` | Exploratory: coherence of rectified EMG within vs between synergy sectors at matched distance, crosstalk controls | [Open in Colab](https://colab.research.google.com/github/eljoserass/semg2pose/blob/main/notebooks/intermuscular_coherence.ipynb) |
| `network_internals.ipynb` | Exploratory: dimensionality, readout and alignment of the pretrained vemg2pose encoder features (treated as a neural population) | [Open in Colab](https://colab.research.google.com/github/eljoserass/semg2pose/blob/main/notebooks/network_internals.ipynb) |

`pipeline_overview`, `compression_tolerance` and `network_internals` also download Meta's pretrained checkpoints (~250 MB) to
`~/emg2pose_model_checkpoints` (helpers in `src/semg2pose/model.py`).

## License

Our code: TBD. emg2pose and its dataset are released by Meta under CC BY-NC-SA 4.0; see
`emg2pose/LICENSE`.
