"""
Build SemiLT/scUSALT inputs for human CARE (GSE165838 RNA + GSE165837 ATAC):
  - rna.h5ad          : common genes, obs['cell_type'] required
  - atac.h5ad         : Signac GeneActivity GAM on common genes, obs['cell_type'] required
                        (cells whose cell_type is absent from RNA are dropped)
  - atac_pca50.h5ad   : 50-dim peak embedding, rows align with ATAC

Example:
  python build_heart_semilt_inputs.py --care-dir unpaird/heart --out-dir unpaird/heart/processed
  # Requires R: Seurat, Signac, rtracklayer. Use --python-gam for legacy Python GAM.
"""

from __future__ import annotations

import argparse
import gzip
import re
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd
import scanpy as sc
import scipy.io
import scipy.sparse as sp
from sklearn.decomposition import TruncatedSVD

from signac_gam import build_gene_activity_python, build_gene_activity_signac

GENCODE_URL = (
    "https://ftp.ebi.ac.uk/pub/databases/gencode/Gencode_human/release_43/"
    "gencode.v43.basic.annotation.gtf.gz"
)
ATAC_LABELS_URL = (
    "https://ftp.ncbi.nlm.nih.gov/geo/series/GSE165nnn/GSE165837/suppl/"
    "GSE165837_CARE_ATAC_merged_umap_cluster_labels.tsv.gz"
)


def _read_barcodes_gz(path: Path) -> list[str]:
    barcodes = []
    with gzip.open(path, "rt", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.replace('"', "").split("\t")
            if parts[0].lower() in {"x", "barcode", "cell"} and len(parts) == 1:
                continue
            barcodes.append(parts[-1] if len(parts) > 1 else parts[0])
    return barcodes


def _read_peak_names_gz(path: Path) -> list[str]:
    with gzip.open(path, "rt", encoding="utf-8") as f:
        return [line.strip().strip('"') for line in f if line.strip()]


def load_mtx(prefix: Path, var_names: list[str], barcodes: list[str]) -> sc.AnnData:
    mtx = prefix.parent / f"{prefix.name}.mtx.gz"
    if not mtx.exists():
        mtx = Path(str(prefix) + ".mtx.gz")
    mat = scipy.io.mmread(mtx)
    if mat.shape[0] == len(var_names):
        mat = mat.T.tocsr()
    else:
        mat = sp.csr_matrix(mat)
    if mat.shape[0] != len(barcodes):
        raise ValueError(f"Barcode count {len(barcodes)} != matrix rows {mat.shape[0]} for {prefix}")
    if mat.shape[1] != len(var_names):
        raise ValueError(f"Feature count {len(var_names)} != matrix cols {mat.shape[1]} for {prefix}")
    return sc.AnnData(X=mat, obs=pd.DataFrame(index=barcodes), var=pd.DataFrame(index=var_names))


def _ensure_cell_type(adata: sc.AnnData, source_cols: tuple[str, ...], name: str) -> None:
    if "cell_type" in adata.obs.columns and adata.obs["cell_type"].notna().all():
        adata.obs["cell_type"] = adata.obs["cell_type"].astype(str)
        return
    for col in source_cols:
        if col in adata.obs.columns:
            adata.obs["cell_type"] = adata.obs[col].astype(str)
            return
    raise ValueError(f"{name} is missing cell_type (tried columns: {source_cols})")


def download_file(url: str, dest: Path) -> Path:
    dest.parent.mkdir(parents=True, exist_ok=True)
    if not dest.exists():
        print(f"Downloading {dest.name} ...")
        urllib.request.urlretrieve(url, dest)
    return dest


def load_care_rna(care_dir: Path) -> sc.AnnData:
    barcodes = _read_barcodes_gz(care_dir / "GSE165838_CARE_RNA_barcodes.txt.gz")
    feat_df = pd.read_csv(care_dir / "GSE165838_CARE_RNA_features.tsv.gz", sep="\t")
    if feat_df.shape[1] >= 2:
        gene_col = feat_df.columns[1] if feat_df.columns[0].lower() in {"x", "id"} else feat_df.columns[-1]
        var_names = feat_df[gene_col].astype(str).tolist()
    else:
        var_names = feat_df.iloc[:, 0].astype(str).tolist()
    adata = load_mtx(care_dir / "GSE165838_CARE_RNA_counts", var_names, barcodes)
    meta = pd.read_csv(care_dir / "GSE165838_CARE_RNA_metadata.txt.gz", sep="\t", index_col=0)
    adata.obs = adata.obs.join(meta, how="left")
    _ensure_cell_type(adata, ("celltype", "cell_type", "Cluster"), "RNA")
    adata.obs["cell_type"] = _normalize_cell_type_labels(adata.obs["cell_type"])
    missing = int(adata.obs["cell_type"].isna().sum())
    if missing:
        raise ValueError(f"RNA has {missing} cells without cell_type after metadata merge.")
    return adata


def _normalize_cell_type_labels(series: pd.Series) -> pd.Series:
    out = series.astype(str).str.strip()
    invalid = {"", "nan", "NaN", "None", "none"}
    return out.where(~out.isin(invalid))


def load_care_atac_peak(care_dir: Path, download_labels: bool) -> sc.AnnData:
    labels_path = care_dir / "GSE165837_CARE_ATAC_merged_umap_cluster_labels.tsv.gz"
    if not labels_path.exists() and download_labels:
        download_file(ATAC_LABELS_URL, labels_path)
    if not labels_path.exists():
        raise FileNotFoundError(
            f"ATAC labels missing: {labels_path.name}\n"
            "Download GSE165837_CARE_ATAC_merged_umap_cluster_labels.tsv.gz from GSE165837, "
            "or rerun with --download-atac-labels."
        )

    barcodes = _read_barcodes_gz(care_dir / "GSE165837_CARE_ATAC_merged_barcodes.txt.gz")
    peaks = _read_peak_names_gz(care_dir / "GSE165837_CARE_ATAC_merged_features.txt.gz")
    adata = load_mtx(care_dir / "GSE165837_CARE_ATAC_merged_matrix", peaks, barcodes)

    labels = pd.read_csv(labels_path, sep="\t", index_col=0)
    adata.obs = adata.obs.join(labels, how="left")
    _ensure_cell_type(adata, ("Cluster", "celltype", "cell_type"), "ATAC")
    adata.obs["cell_type"] = _normalize_cell_type_labels(adata.obs["cell_type"])

    n_missing = int(adata.obs["cell_type"].isna().sum())
    if n_missing:
        print(f"Dropping {n_missing} ATAC cells without labels (not in label file).")
        adata = adata[adata.obs["cell_type"].notna()].copy()
    adata.obs["cell_type"] = adata.obs["cell_type"].astype(str)
    return adata


def tfidf_sparse(X: sp.spmatrix) -> sp.csr_matrix:
    X = X.tocsr().astype(np.float32)
    cell_sums = np.asarray(X.sum(axis=1)).ravel()
    cell_sums[cell_sums == 0] = 1.0
    tf = X.multiply(1.0 / cell_sums[:, None])
    df = np.asarray((X > 0).sum(axis=0)).ravel()
    idf = np.log1p(X.shape[0] / (1.0 + df)).astype(np.float32)
    return tf.multiply(idf).tocsr()


def peak_pca(peak_adata: sc.AnnData, n_components: int = 50) -> sc.AnnData:
    X = peak_adata.X if sp.issparse(peak_adata.X) else sp.csr_matrix(peak_adata.X)
    X_tfidf = tfidf_sparse(X)
    n_components = min(n_components, X_tfidf.shape[0] - 1, X_tfidf.shape[1])
    emb = TruncatedSVD(n_components=n_components, random_state=1).fit_transform(X_tfidf)
    out = sc.AnnData(
        X=emb.astype(np.float32),
        obs=peak_adata.obs.copy(),
        var=pd.DataFrame(index=[f"feat_{i}" for i in range(n_components)]),
    )
    out.obs_names = peak_adata.obs_names.astype(str)
    return out


def align_common_genes(rna: sc.AnnData, atac: sc.AnnData) -> tuple[sc.AnnData, sc.AnnData]:
    common = rna.var_names.astype(str).intersection(atac.var_names.astype(str)).sort_values()
    if len(common) == 0:
        raise ValueError("No common genes between RNA and ATAC.")
    rna = rna[:, common].copy()
    atac = atac[:, common].copy()
    return rna, atac


def filter_atac_to_rna_cell_types(
    rna: sc.AnnData, atac: sc.AnnData, pca50: sc.AnnData
) -> tuple[sc.AnnData, sc.AnnData]:
    """Keep ATAC cells whose cell_type name appears in the RNA reference."""
    rna_types = set(rna.obs["cell_type"].astype(str))
    atac_labels = atac.obs["cell_type"].astype(str)
    only_atac = sorted(set(atac_labels) - rna_types)
    mask = atac_labels.isin(rna_types).values
    n_drop = int((~mask).sum())
    if n_drop:
        print(
            f"ATAC: dropped {n_drop} cells with cell_type not in RNA "
            f"({len(only_atac)} ATAC-only type(s): {only_atac})"
        )
    atac = atac[mask].copy()
    if atac.n_obs == 0:
        raise ValueError("No ATAC cells left after filtering to RNA cell types.")
    pca50 = pca50[atac.obs_names].copy()
    return atac, pca50


def validate_outputs(rna: sc.AnnData, atac: sc.AnnData, pca50: sc.AnnData) -> None:
    if "cell_type" not in rna.obs or rna.obs["cell_type"].isna().any():
        raise ValueError("rna.h5ad must have cell_type for every cell.")
    if "cell_type" not in atac.obs or atac.obs["cell_type"].isna().any():
        raise ValueError("atac.h5ad must have cell_type for every cell.")
    if rna.n_vars != atac.n_vars or not rna.var_names.equals(atac.var_names):
        raise ValueError("RNA and ATAC gene sets are not aligned.")
    if pca50.n_obs != atac.n_obs or not pca50.obs_names.equals(atac.obs_names):
        raise ValueError("atac_pca50 rows must match ATAC cells.")
    if pca50.n_vars != 50:
        raise ValueError(f"atac_pca50 must have 50 dimensions, got {pca50.n_vars}.")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--care-dir", type=Path, default=Path("unpaird/heart"))
    parser.add_argument("--out-dir", type=Path, default=Path("unpaird/heart/processed"))
    parser.add_argument("--gtf", type=Path, default=None)
    parser.add_argument("--download-gtf", action="store_true")
    parser.add_argument("--download-atac-labels", action="store_true")
    parser.add_argument("--n-pca", type=int, default=50)
    parser.add_argument(
        "--python-gam",
        action="store_true",
        help="Use legacy Python GAM instead of Signac GeneActivity()",
    )
    args = parser.parse_args()

    rna = load_care_rna(args.care_dir)
    peak = load_care_atac_peak(args.care_dir, download_labels=args.download_atac_labels)

    gtf = args.gtf
    if gtf is None and args.download_gtf:
        gtf = download_file(GENCODE_URL, args.out_dir / "cache" / "gencode.v43.basic.annotation.gtf.gz")
    if gtf is None:
        gtf = args.out_dir / "cache" / "gencode.v43.basic.annotation.gtf.gz"
    if not Path(gtf).exists():
        raise SystemExit("GTF not found. Use --download-gtf or pass --gtf.")

    gtf_path = Path(gtf)
    if args.python_gam:
        print("Computing gene activity from peaks (Python GAM) ...")
        atac = build_gene_activity_python(peak, gtf_path)
    else:
        print("Computing gene activity with Signac GeneActivity() ...")
        atac = build_gene_activity_signac(
            peak, gtf_path=gtf_path, genome="hg38", cache_dir=args.out_dir / "cache"
        )
    _ensure_cell_type(atac, ("cell_type", "Cluster"), "ATAC GAM")

    print("Peak TF-IDF + TruncatedSVD (50 dims) ...")
    pca50 = peak_pca(peak, n_components=args.n_pca)

    print("Filtering ATAC to cell types present in RNA ...")
    atac, pca50 = filter_atac_to_rna_cell_types(rna, atac, pca50)

    rna, atac = align_common_genes(rna, atac)
    validate_outputs(rna, atac, pca50)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    rna.write_h5ad(args.out_dir / "rna.h5ad", compression="gzip")
    atac.write_h5ad(args.out_dir / "atac.h5ad", compression="gzip")
    pca50.write_h5ad(args.out_dir / "atac_pca50.h5ad", compression="gzip")

    print(f"rna.h5ad:       {rna.n_obs} x {rna.n_vars}, cell_type={rna.obs['cell_type'].nunique()} types")
    print(f"atac.h5ad:      {atac.n_obs} x {atac.n_vars}, cell_type={atac.obs['cell_type'].nunique()} types")
    print(f"atac_pca50.h5ad:{pca50.n_obs} x {pca50.n_vars}")
    shared = set(rna.obs["cell_type"]) & set(atac.obs["cell_type"])
    print(f"  Shared type names (eval): {len(shared)} -> {sorted(shared)}")


if __name__ == "__main__":
    main()
