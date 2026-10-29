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

Outputs in `output_scUSALT/`:

- `*_embeddings.txt` — RNA / ATAC latent vectors
- `*_predictions.txt` — one predicted cell type per ATAC cell (argmax)
- `*_type_prediction_confidence.tsv` — long table (`a × N` rows): `cell_index`, `cell_type`, `confidence_score` (0.5×K + 0.5×D for that type)

## Data

| File | Role |
|------|------|
| RNA `.h5ad` | Reference; needs `obs['cell_type']` |
| ATAC `.h5ad` | Query gene-activity matrix (same genes as RNA) |
| Peak `.h5ad` | ATAC peak/PCA features (for contrastive loss) |
| Protein `.h5ad` | Optional (e.g. CITE/ASAP) |

`input_size` in `setting.py` must match `adata.X.shape[1]`.
