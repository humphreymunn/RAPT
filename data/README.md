# Data

RAPT consumes **sequence datasets**: N sequences, each `[T_i, D]` (time ×
dimensions), where `T_i` may differ between sequences. The signal can be
anything temporal — robot proprioception, tabular sensor streams, or a
learned latent space. See the docstring of `rapt/data.py` for the accepted
on-disk formats (.npz, .h5, directory of .csv/.npy) and the optional
`actions` / `labels` / `dim_names` side-channels.

`data/sample/` holds a small synthetic dataset for the quickstart; regenerate
or resize it with:

```bash
python scripts/make_synthetic_data.py --out data/sample
```

The real-robot Unitree G1 logs (78 labeled runs) and the Isaac Lab collection
outputs used in the paper are not stored in this repository — see
`reproduce/README.md` for how they were collected and their exact CSV/HDF5
formats.
