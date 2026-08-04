# scUSALT

scUSALT is a structure- and alignment-aware framework for cross-modal label transfer from scRNA-seq to scATAC-seq. It learns a shared latent space via embedding regularization, cross-modal anchor alignment, domain-adversarial alignment, graph contrastive learning on peak-derived ATAC neighborhoods, and RNA reference-label supervision, then transfers cell-type annotations to unlabeled scATAC-seq cells.

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
