# scUSALT

Semi-supervised **label transfer from scRNA-seq → scATAC-seq**.

Learns a shared embedding with MNN alignment, **graph contrastive learning** (ATAC peaks), and **domain-adversarial alignment**, then transfers labels by kNN.

## Setup

```bash
# Python 3.8+, PyTorch, scanpy, anndata, numpy, scipy, scikit-learn
```

## Run

1. Put `.h5ad` files under `Data/` (paths in `setting.py`).
2. Edit `setting.py`:

```python
DB = 'CITE_ASAP'       # dataset name
self.use_cuda = False  # True for GPU
```

3. Train + transfer:

```bash
python main.py
```

Outputs in `output_scUSALT/`: `*_embeddings.txt`, `*_predictions.txt`.

## Data

| File | Role |
|------|------|
| RNA `.h5ad` | Reference; needs `obs['cell_type']` |
| ATAC `.h5ad` | Query gene-activity matrix (same genes as RNA) |
| Peak `.h5ad` | ATAC peak/PCA features (for contrastive loss) |
| Protein `.h5ad` | Optional (e.g. CITE/ASAP) |

`input_size` in `setting.py` must match `adata.X.shape[1]`.
