# HGPTrans: Hierarchical Graph-Pooling Transolver for Vehicle Drag Prediction

> Official PyTorch implementation of HGPTrans: hierarchical graph-pooling Transolver for vehicle aerodynamic drag prediction (Cd regression) from STL meshes.

Predict the aerodynamic drag coefficient (Cd) of a vehicle directly from its
surface mesh. The model builds a graph from the triangle mesh (vertices +
normals as node features, mesh edges as the graph edge set), runs GIN message
passing, applies Transolver-style physics attention over learned slice tokens
(suited to irregular meshes), and performs hierarchical TopK graph pooling
with a multi-scale pooled readout.

<!-- Optional: add a pipeline figure at assets/pipeline.png and uncomment
<p align="center">
  <img src="assets/pipeline.png" width="720" alt="pipeline">
</p>
-->

## Repository structure

```
├── models.py        # HGPTrans model (GIN message passing + physics attention)
├── dataset.py       # STL mesh-graph dataset + per-sample standardization
├── train.py         # DDP training loop (gradient accumulation, checkpointing)
├── evaluate.py      # Test-split / extra-STL evaluation with metrics
├── utils.py         # logging, label lookup, seeding, plotting helpers
├── requirements.txt
```

## Installation

Requires Python 3.9+ and CUDA (for training on GPU).

```bash
git clone https://github.com/BoLiu-USTC/HGPTrans.git
cd HGPTrans
pip install -r requirements.txt
```

`torch` and `torch_geometric` are listed but their CUDA-matched wheels depend
on your local CUDA version, so install them first following the official
guides if the pinned wheels do not match your driver:

- PyTorch: https://pytorch.org/get-started/locally/
- PyG: https://pytorch-geometric.readthedocs.io/en/latest/install/installation.html

## Data layout

Point `--data_dir` at a dataset directory with this layout
(DrivAerNet / DrivAerNet++ style):

```
data_dir/
├── cleaned_stls/               # one <design_id>.stl per design
├── labels.xlsx                 # columns: ID, Drag_Value
└── train_val_test_splits/
    ├── train_design_ids.txt    # one design id per line
    ├── val_design_ids.txt
    └── test_design_ids.txt
```

The DrivAerNet++ 8k+ variant (`train.txt` / `val.txt` / `test.txt` splits and
`DrivAerNetPlusPlus_Cd_8k_Updated.xlsx`) is auto-detected as well.

## Training

```bash
# single GPU
python train.py --data_dir /path/to/DrivAerNet++ --save_dir ./logs/run1

# multi GPU (DDP)
python train.py --data_dir /path/to/DrivAerNet++ --save_dir ./logs/run1 --world_size 4

# resume from <save_dir>/checkpoint.pth
python train.py --data_dir /path/to/DrivAerNet++ --save_dir ./logs/run1 --resume
```

Key options: `--accumulation_steps`
(default 16, i.e. effective batch = batch_size × accumulation_steps =
16 × 1). Vertex features are standardized per sample (zero mean / unit
variance over each mesh).

## Evaluation

```bash
# on the dataset test split
python evaluate.py --data_dir /path/to/DrivAerNet++ \
    --checkpoint ./logs/run1/model_best.pt

# on a folder of extra STL files whose names encode the true Cd,
# e.g. design_0.312.stl -> truth Cd = 0.312
python evaluate.py --data_dir /path/to/DrivAerNet++ \
    --checkpoint ./logs/run1/model_best.pt --extra_data ./data/Ahmed
```

Per-design predictions are written to `predLog/preds.txt` next to the
checkpoint; metrics (MSE / MAE / R² / rel-L2 / Pearson / Spearman) are printed
and an error-scatter plot is saved to `predLog/error.png`.

## Notes

- Keep `--batch_size 1`: model forwards use dense reshapes that assume a
  single graph per batch. Use `--accumulation_steps` for larger effective
  batches.
- All meshes above `--simplify_size` vertices (default 417,114) are decimated
  with quadric edge collapse before graph construction.
- Training latency printed during validation is forward-only; the end-to-end
  per-car inference time (mesh loading + QEM decimation + forward) reported in
  the paper is measured over the full pipeline.

## Citation

If you use this code, please cite:

```bibtex
@article{liu2026hgptrans,
  title  = {HGPTrans: Hierarchical Graph-Pooling Transolver for Automotive Aerodynamic Drag Prediction},
  author = {Bo Liu and others},
  journal= {<Journal>},
  year   = {2026}
}
```

## License

Released under the [MIT License](LICENSE).
