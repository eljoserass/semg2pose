"""Reading emg2pose session files directly with h5py (no PyTorch dataset needed)."""

import tarfile
import urllib.request
from pathlib import Path

import h5py
import numpy as np
import pandas as pd

FS = 2000  # EMG and joint-angle sampling rate (Hz)
N_CHANNELS = 16

DEFAULT_DOWNLOAD_DIR = Path.home()
DEFAULT_DATASET_DIR = DEFAULT_DOWNLOAD_DIR / "emg2pose_dataset_mini"
MINI_DATASET_URL = "https://fb-ctrl-oss.s3.amazonaws.com/emg2pose/emg2pose_dataset_mini.tar"

# Order of the 20 joint angles, as in emg2pose/constants.py
# (FE = flexion/extension, AA = abduction/adduction)
JOINT_NAMES = [
    "THUMB_CMC_FE", "THUMB_CMC_AA", "THUMB_MCP_FE", "THUMB_IP_FE",
    "INDEX_MCP_AA", "INDEX_MCP_FE", "INDEX_PIP_FE", "INDEX_DIP_FE",
    "MIDDLE_MCP_AA", "MIDDLE_MCP_FE", "MIDDLE_PIP_FE", "MIDDLE_DIP_FE",
    "RING_MCP_AA", "RING_MCP_FE", "RING_PIP_FE", "RING_DIP_FE",
    "PINKY_MCP_AA", "PINKY_MCP_FE", "PINKY_PIP_FE", "PINKY_DIP_FE",
]
# MCP + PIP flexion of fingers 2-5: their mean is "how closed the hand is"
FINGER_FLEXION = [5, 6, 9, 10, 13, 14, 17, 18]


def download_mini_dataset(download_dir=DEFAULT_DOWNLOAD_DIR):
    """Download and unpack the ~600 MiB mini dataset if needed. Returns the dataset folder."""
    download_dir = Path(download_dir)
    dataset_dir = download_dir / "emg2pose_dataset_mini"
    if dataset_dir.exists():
        print(f"Dataset already downloaded: {dataset_dir}")
        return dataset_dir
    download_dir.mkdir(parents=True, exist_ok=True)
    tar_path = download_dir / "emg2pose_dataset_mini.tar"
    print("Downloading ~600 MiB (takes a few minutes)...")
    urllib.request.urlretrieve(MINI_DATASET_URL, tar_path)
    with tarfile.open(tar_path) as tar:
        tar.extractall(download_dir, filter="data")
    tar_path.unlink()
    print(f"Done: {dataset_dir}")
    return dataset_dir


def load_session(path):
    """Return emg (T, 16), angles (T, 20) in radians, tracking-ok mask (T,) and metadata."""
    with h5py.File(path, "r") as f:
        group = f["emg2pose"]
        ts = group["timeseries"][:]  # structured array with fields: time, joint_angles, emg
        meta = dict(group.attrs)
    emg = ts["emg"].astype(np.float64)
    angles = ts["joint_angles"].astype(np.float64)
    tracking_ok = ~np.all(np.isclose(angles, 0), axis=1)  # all-zero rows = IK failure
    return emg, angles, tracking_ok, meta


def list_sessions(dataset_dir=DEFAULT_DATASET_DIR):
    """One row per .hdf5 file: path, stage, side, user, seconds (reads metadata only)."""
    rows = []
    for path in sorted(Path(dataset_dir).glob("*.hdf5")):
        with h5py.File(path, "r") as f:
            g = f["emg2pose"]
            rows.append({"path": path, "stage": g.attrs["stage"], "side": g.attrs["side"],
                         "user": g.attrs["user"], "seconds": len(g["timeseries"]) / FS})
    return pd.DataFrame(rows)


def find_session(stage, side="right", sessions=None, dataset_dir=DEFAULT_DATASET_DIR):
    """Path of the first recording of a given stage and hand side."""
    if sessions is None:
        sessions = list_sessions(dataset_dir)
    match = sessions[(sessions.stage == stage) & (sessions.side == side)]
    if match.empty:
        raise KeyError(f"No {side}-hand recording of stage {stage!r}")
    return match.path.iloc[0]
