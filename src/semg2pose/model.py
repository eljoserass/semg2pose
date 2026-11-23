"""Loading Meta's pretrained emg2pose checkpoints and running them on whole sessions."""

import tarfile
import urllib.request
from pathlib import Path

import numpy as np
import torch

CHECKPOINTS_URL = "https://fb-ctrl-oss.s3.amazonaws.com/emg2pose/emg2pose_model_checkpoints.tar.gz"
DEFAULT_CHECKPOINT_DIR = Path.home() / "emg2pose_model_checkpoints"
METADATA_URL = "https://fb-ctrl-oss.s3.amazonaws.com/emg2pose/emg2pose_metadata.csv"


def download_checkpoints(download_dir=Path.home()):
    """Download and unpack the ~250 MB checkpoint archive if needed. Returns the folder."""
    download_dir = Path(download_dir)
    ckpt_dir = download_dir / "emg2pose_model_checkpoints"
    if ckpt_dir.exists():
        print(f"Checkpoints already downloaded: {ckpt_dir}")
        return ckpt_dir
    tar_path = download_dir / "emg2pose_model_checkpoints.tar.gz"
    print("Downloading ~250 MB of checkpoints...")
    urllib.request.urlretrieve(CHECKPOINTS_URL, tar_path)
    with tarfile.open(tar_path) as tar:
        tar.extractall(download_dir, filter="data")
    tar_path.unlink()
    return ckpt_dir


def download_metadata(download_dir=Path.home()):
    """Download the 5 MB metadata CSV of the full dataset (one row per recording) if needed."""
    path = Path(download_dir) / "emg2pose_metadata.csv"
    if not path.exists():
        urllib.request.urlretrieve(METADATA_URL, path)
    return path


def load_pretrained(experiment="tracking_vemg2pose", ckpt_dir=DEFAULT_CHECKPOINT_DIR):
    """Load a pretrained Emg2PoseModule (CPU, eval mode). experiment must match the checkpoint name."""
    from emg2pose.lightning import Emg2PoseModule
    from emg2pose.utils import generate_hydra_config_from_overrides

    ckpt = Path(ckpt_dir) / f"{experiment}.ckpt"
    cfg = generate_hydra_config_from_overrides(
        overrides=[f"experiment={experiment}", f"checkpoint={ckpt}"])
    module = Emg2PoseModule.load_from_checkpoint(
        cfg.checkpoint, network=cfg.network, optimizer=cfg.optimizer,
        lr_scheduler=cfg.lr_scheduler, map_location="cpu")
    return module.eval()


def predict_session(module, path, window_length=11_790, step=40):
    """Run a module over every IK-failure-free window of a session, like the benchmark's val/test loop.

    Windows are 11,790 samples: the first 1,790 are left context (no prediction), the next
    10,000 (5 s) are predicted. Returns (pred, target) in radians, shape (N, 20), keeping
    one sample out of `step` (40 -> 50 Hz), or (None, None) if the session has no full window.
    """
    from emg2pose.data import WindowedEmgDataset

    ds = WindowedEmgDataset(path, window_length=window_length, skip_ik_failures=True)
    if len(ds) == 0:
        return None, None
    batch = torch.utils.data.default_collate([ds[i] for i in range(len(ds))])
    batch["no_ik_failure"] = module.update_ik_failure_mask(batch["no_ik_failure"])
    with torch.no_grad():
        pred, target, mask = module.forward(batch)
    pred = pred.permute(0, 2, 1)[mask].numpy()[::step]
    target = target.permute(0, 2, 1)[mask].numpy()[::step]
    return pred.astype(np.float64), target.astype(np.float64)
