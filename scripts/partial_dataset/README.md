# Downloading part of the full emg2pose dataset

`notebooks/cross_user_invariance.ipynb` uses 12 users (52 recordings each, 11.5 GB) from the full 431 GiB tar,
without downloading the whole archive:

```shell
# 1. Index the remote tar (name, offset, size of every member) with HTTP range requests (~10 MB of traffic)
uv run python scripts/partial_dataset/tar_index.py tar_index.json
# 2. Download only the members listed in wanted.txt (the 624 recordings used in the notebook)
uv run python scripts/partial_dataset/fetch_members.py tar_index.json scripts/partial_dataset/wanted.txt ~/emg2pose_dataset_partial 8
```

Edit `wanted.txt` (one `.hdf5` file name per line, see `~/emg2pose_metadata.csv`) to fetch other users or stages.
