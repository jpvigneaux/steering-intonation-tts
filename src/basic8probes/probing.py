"""Probe models, feature extractors and the training loop.

This is the core of the probing pipeline: every driver script in ``pipeline/`` is a wrapper
around the classes defined here.

The central design decision is the boundary between two families of read-out:

* Most extractors reduce a variable-length ``[D, N]`` frame trajectory to a fixed-size vector
  *on the data side*, before the classifier sees it. They subclass :class:`FeatureExtractor`.
* The extractors named in :data:`SEQUENCE_EXTRACTOR_TYPES` are the exception. Their temporal
  aggregation lives *inside* the model and is trained jointly with the classifier, so the data
  side returns ``[N, k]`` sequences batched by :func:`pad_collate` and the probe's ``forward``
  takes ``(padded, mask)``.

A new read-out must pick a side of that boundary; conflating the two is the likeliest way to
break the pipeline.

Only path signatures and functional PCA preserve time structure. Learnable weighting and
attention pooling are weighted averages: whatever the weights, an average is not invertible
back to the trajectory, so a rise followed by a fall is intrinsically hard to distinguish from
a fall followed by a rise.
"""

from abc import ABC, abstractmethod
from datetime import datetime, timezone
import math
import os
import tempfile
import dataclasses
from dataclasses import dataclass
import json
import joblib
from pathlib import Path

import pandas as pd
from sklearn.decomposition import PCA

import torch
from torch.utils.data import DataLoader, Dataset
import torch.nn as nn
import torch.optim as optim

import torch.distributed as dist
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group
import iisignature

import optuna
import typing

@dataclass
class HyperparametersProbe:
    pca_dim: int
    whiten: bool
    batch_size: int
    learning_rate: float
    decay_rate_end: float
    n_epochs: int
    decay_rate_beg: float = 0
    hidden_dim_multiplier: float = 0.0  # When = 0.0, defaults to a linear probe; if > 0 nonlinear
    extractor_type: str = "weighted_mean_pca"  # weighted_mean_pca | path_signature | fpca |
                                               # lda_mean | learnable_weighting | attention_pooling
    sig_depth: int = 3          # path signature truncation depth (PathSignatureExtractor only)
    pre_sig_pca_dim: int = 16   # channel PCA dim before signature (PathSignatureExtractor only)
    fpca_n_basis: int = 8       # number of order-4 B-spline basis functions (FunctionalPCAExtractor only)
    fpca_var_threshold: float = 0.95  # cumulative-variance cutoff tau for FPC scores (FunctionalPCAExtractor only)
    weighting_knots: int = 5    # K interpolation knots of the weighting profile (LearnableWeightingProbe only)
    num_queries: int = 1        # number of independent attention queries (AttentionPoolingProbe only)
    pe_bands: int = 6           # sinusoidal positional-encoding bands added to attention keys (AttentionPoolingProbe only)
    class_weighting: str = "none"  # none | balanced | balanced_by_tune -- see ProbeTrainer
    train_prefix_augment: float = 0.0  # P(truncate an example to a random prefix) per epoch;
                                       # 0.0 = off, which is every run before 2026-08-13.
    prefix_augment_min: float = 0.1    # shortest prefix sampled, as a fraction of the sequence


# Extractor types whose temporal aggregation lives inside the probe model (trained jointly
# with the classifier); their data-side extractor returns [N, k] sequences, batched with
# pad_collate, and the probe forward takes (padded, mask).
SEQUENCE_EXTRACTOR_TYPES = {"learnable_weighting", "attention_pooling"}

@dataclass
class RunData:
    n_runs: int
    hyperparam_key: tuple
    best_mean_accuracy: float


def get_timestamp()->str:
    timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%d_%H-%M-%S")
    return timestamp

# ---------------------------------------------------------------------------
# Internal PCA helper (used by feature extractors)
# ---------------------------------------------------------------------------

class PCATransformer:
    def __init__(self, n_components = 64, whiten = False):
        self.model = PCA(n_components=n_components, whiten=whiten)
        self.fitted = False

    def fit(self, features):
        """features: torch.Tensor [N, D]"""
        self.model.fit(features.cpu().numpy())
        self.fitted = True

    def transform(self, features):
        """Transform a tensor of shape [D] or [N, D]. Returns torch.Tensor."""
        X = features.cpu().numpy()
        X_t = self.model.transform(X)
        return torch.from_numpy(X_t).float()

    def save(self, path):
        joblib.dump(self.model, path)

    @classmethod
    def load(cls, path):
        obj = cls()
        obj.model = joblib.load(path)
        obj.fitted = True
        return obj

# ---------------------------------------------------------------------------
# Feature extractor abstraction
# ---------------------------------------------------------------------------

class FeatureExtractor(ABC):
    """Converts a variable-length [D, N] embedding tensor into a fixed-size 1-D feature vector.

    Exception: extractors with returns_sequence = True output a [N, feature_dim] sequence
    instead; temporal aggregation then happens inside the probe model (see pad_collate and
    the sequence probes)."""

    returns_sequence: bool = False

    @abstractmethod
    def fit(self, embeddings: list[torch.Tensor], labels: list | None = None) -> None:
        """Fit any learnable components (e.g. PCA) on a list of [D, N] training tensors.

        labels: training labels, aligned with embeddings; only supervised extractors
        (LDAMeanExtractor) use them, all others ignore the argument."""
        ...

    @abstractmethod
    def transform(self, emb: torch.Tensor) -> torch.Tensor:
        """Transform a single [D, N] tensor into a 1-D feature vector of length feature_dim
        (or a [N, feature_dim] sequence if returns_sequence)."""
        ...

    @property
    @abstractmethod
    def feature_dim(self) -> int:
        """Output dimensionality after transform."""
        ...

    @abstractmethod
    def save(self, path: Path) -> None:
        ...

    @classmethod
    @abstractmethod
    def _from_dict(cls, data: dict) -> "FeatureExtractor":
        """Reconstruct from the dict persisted by save()."""
        ...


class WeightedMeanPCAExtractor(FeatureExtractor):
    """Temporal weighted mean → PCA (the original approach)."""

    def __init__(self, pca_dim: int, whiten: bool, decay_rate_end: float, decay_rate_beg: float = 0.0):
        self.decay_rate_end = decay_rate_end
        self.decay_rate_beg = decay_rate_beg
        self.pca = PCATransformer(n_components=pca_dim, whiten=whiten)

    def _temporal_weighted_mean(self, emb: torch.Tensor) -> torch.Tensor:
        """[D, N] → [D]"""
        N = emb.shape[1]
        weights = (torch.exp(-self.decay_rate_end * torch.arange(N - 1, -1, -1, dtype=torch.float))
                   + torch.exp(-self.decay_rate_beg * torch.arange(N, dtype=torch.float)))
        weights = weights / weights.sum()
        return (emb * weights).sum(dim=1)

    def fit(self, embeddings: list[torch.Tensor], labels: list | None = None) -> None:
        means = torch.stack([self._temporal_weighted_mean(e) for e in embeddings], dim=0)  # [N_train, D]
        self.pca.fit(means)

    def transform(self, emb: torch.Tensor) -> torch.Tensor:
        mean = self._temporal_weighted_mean(emb)                    # [D]
        return self.pca.transform(mean.unsqueeze(0)).squeeze(0)     # [pca_dim]

    @property
    def feature_dim(self) -> int:
        return self.pca.model.n_components

    def save(self, path: Path) -> None:
        joblib.dump({
            "type": "weighted_mean_pca",
            "decay_rate_end": self.decay_rate_end,
            "decay_rate_beg": self.decay_rate_beg,
            "pca_model": self.pca.model,
        }, path)

    @classmethod
    def _from_dict(cls, data: dict) -> "WeightedMeanPCAExtractor":
        pca_dim = data["pca_model"].n_components
        whiten  = data["pca_model"].whiten
        obj = cls(pca_dim, whiten, data["decay_rate_end"], data["decay_rate_beg"])
        obj.pca.model  = data["pca_model"]
        obj.pca.fitted = True
        return obj


class PathSignatureExtractor(FeatureExtractor):
    """Truncated path signature of a PCA-reduced linear interpolation of [D, N] embeddings.

    Steps: PCA along channel axis D → pre_sig_pca_dim; transpose to [N, pre_sig_pca_dim];
    compute iisignature.sig(path, sig_depth).
    Feature dimension: iisignature.siglength(pre_sig_pca_dim, sig_depth).
    """

    def __init__(self, pre_sig_pca_dim: int, sig_depth: int, whiten: bool):
        self.pre_sig_pca_dim = pre_sig_pca_dim
        self.sig_depth = sig_depth
        self.whiten = whiten
        self.pca = PCATransformer(n_components=pre_sig_pca_dim, whiten=self.whiten)
        self._feature_dim: int | None = None

    def fit(self, embeddings: list[torch.Tensor], labels: list | None = None) -> None:
        # Fit PCA on all individual frames across the training set
        all_frames = torch.cat([e.T for e in embeddings], dim=0)  # [total_frames, D]
        self.pca.fit(all_frames)
        self._feature_dim = iisignature.siglength(self.pre_sig_pca_dim, self.sig_depth)

    def transform(self, emb: torch.Tensor) -> torch.Tensor:
        path = self.pca.transform(emb.T).numpy()  # [N, pre_sig_pca_dim] as np array
        sig = iisignature.sig(path, self.sig_depth)
        return torch.from_numpy(sig).float()

    @property
    def feature_dim(self) -> int:
        if self._feature_dim is None:
            self._feature_dim = iisignature.siglength(self.pre_sig_pca_dim, self.sig_depth)
        return self._feature_dim

    def save(self, path: Path) -> None:
        joblib.dump({
            "type": "path_signature",
            "pre_sig_pca_dim": self.pre_sig_pca_dim,
            "sig_depth": self.sig_depth,
            "pca_model": self.pca.model,
            "_feature_dim": self._feature_dim,
            "pca_whiten": self.whiten
        }, path)

    @classmethod
    def _from_dict(cls, data: dict) -> "PathSignatureExtractor":
        obj = cls(data["pre_sig_pca_dim"], data["sig_depth"], data.get("pca_whiten", False))
        obj.pca.model   = data["pca_model"]
        obj.pca.fitted  = True
        obj._feature_dim = data["_feature_dim"]
        return obj


class PerFramePCAExtractor(FeatureExtractor):
    """Per-frame PCA only: [D, N] → [N, k], no temporal aggregation.

    Used by the sequence probes (learnable_weighting, attention_pooling), which aggregate
    over time inside the model so that the aggregation weights are trained jointly with the
    classifier. The PCA is fitted on individual frames prior to training, so the model that
    follows it is end-to-end differentiable.
    """

    returns_sequence = True

    def __init__(self, pca_dim: int, whiten: bool):
        self.whiten = whiten
        self.pca = PCATransformer(n_components=pca_dim, whiten=whiten)

    def fit(self, embeddings: list[torch.Tensor], labels: list | None = None) -> None:
        all_frames = torch.cat([e.T for e in embeddings], dim=0)  # [total_frames, D]
        self.pca.fit(all_frames)

    def transform(self, emb: torch.Tensor) -> torch.Tensor:
        return self.pca.transform(emb.T)  # [N, k]

    @property
    def feature_dim(self) -> int:
        return self.pca.model.n_components

    def save(self, path: Path) -> None:
        joblib.dump({
            "type": "per_frame_pca",
            "pca_model": self.pca.model,
            "pca_whiten": self.whiten,
        }, path)

    @classmethod
    def _from_dict(cls, data: dict) -> "PerFramePCAExtractor":
        obj = cls(data["pca_model"].n_components, data.get("pca_whiten", False))
        obj.pca.model = data["pca_model"]
        obj.pca.fitted = True
        return obj


class FunctionalPCAExtractor(FeatureExtractor):
    """Functional PCA of the per-frame-PCA-reduced trajectory.

    Each reduced trajectory [N, k] is a vector-valued function of normalized time
    t ∈ [0,1]: every channel is least-squares-smoothed onto a common order-4 (cubic)
    B-spline basis with n_basis functions, giving coefficients C ∈ R^{n_basis × k}.
    The functional inner product <f, g> = Σ_d ∫ f_d g_d dt equals the Euclidean inner
    product of the Gram-corrected coefficients vec(Lᵀ C) with G = L Lᵀ the basis Gram
    matrix, so FPC scores are obtained by ordinary PCA on those vectors. The number of
    retained components M is set by var_threshold ∈ (0,1) on cumulative explained
    variance, so feature_dim is only known after fit.
    """

    _RIDGE = 1e-6  # regularizes the per-trajectory least squares when N < n_basis

    def __init__(self, pca_dim: int, n_basis: int, var_threshold: float, whiten: bool):
        if n_basis < 4:
            raise ValueError(f"order-4 B-spline basis requires n_basis >= 4, got {n_basis}")
        if not 0.0 < var_threshold < 1.0:
            raise ValueError(f"var_threshold must be in (0,1), got {var_threshold}")
        self.n_basis = n_basis
        self.var_threshold = var_threshold
        self.whiten = whiten
        self.frame_pca = PCATransformer(n_components=pca_dim, whiten=whiten)
        # sklearn PCA with float n_components selects M by cumulative explained variance
        self.score_pca = PCATransformer(n_components=var_threshold, whiten=False)
        self._design_cache: dict[int, "np.ndarray"] = {}
        self._feature_dim: int | None = None
        self._knots, self._gram_chol = self._basis(n_basis)

    @staticmethod
    def _basis(n_basis: int):
        """Order-4 B-spline knot vector on [0,1] and Cholesky factor of the Gram matrix."""
        import numpy as np
        from scipy.interpolate import BSpline
        degree = 3
        n_interior = n_basis - degree - 1
        interior = np.linspace(0, 1, n_interior + 2)[1:-1]
        knots = np.concatenate([np.zeros(degree + 1), interior, np.ones(degree + 1)])
        grid = np.linspace(0.0, 1.0, 2049)
        B = BSpline.design_matrix(grid, knots, degree).toarray()  # [grid, n_basis]
        w = np.full(grid.shape, grid[1] - grid[0])
        w[0] = w[-1] = w[0] / 2  # trapezoid rule
        gram = B.T @ (B * w[:, None])
        L = np.linalg.cholesky(gram + 1e-10 * np.eye(n_basis))
        return knots, L

    def _design(self, N: int):
        """Design matrix of the basis at N equispaced normalized positions (cached per N)."""
        import numpy as np
        from scipy.interpolate import BSpline
        if N not in self._design_cache:
            p = np.linspace(0.0, 1.0, N) if N > 1 else np.array([0.0])
            self._design_cache[N] = BSpline.design_matrix(p, self._knots, 3).toarray()
        return self._design_cache[N]

    def _coefficient_vector(self, emb: torch.Tensor) -> torch.Tensor:
        """[D, N] → Gram-corrected, flattened B-spline coefficients [n_basis * k]."""
        import numpy as np
        Z = self.frame_pca.transform(emb.T).numpy()          # [N, k]
        B = self._design(Z.shape[0])                          # [N, n_basis]
        A = B.T @ B + self._RIDGE * np.eye(self.n_basis)
        C = np.linalg.solve(A, B.T @ Z)                       # [n_basis, k]
        return torch.from_numpy((self._gram_chol.T @ C).flatten()).float()

    def fit(self, embeddings: list[torch.Tensor], labels: list | None = None) -> None:
        all_frames = torch.cat([e.T for e in embeddings], dim=0)
        self.frame_pca.fit(all_frames)
        coeffs = torch.stack([self._coefficient_vector(e) for e in embeddings], dim=0)
        self.score_pca.fit(coeffs)
        self._feature_dim = self.score_pca.model.n_components_

    def transform(self, emb: torch.Tensor) -> torch.Tensor:
        c = self._coefficient_vector(emb)
        return self.score_pca.transform(c.unsqueeze(0)).squeeze(0)  # [M]

    @property
    def feature_dim(self) -> int:
        if self._feature_dim is None:
            raise RuntimeError("FunctionalPCAExtractor.feature_dim is only defined after fit() "
                               "(M is selected by the variance threshold).")
        return self._feature_dim

    def save(self, path: Path) -> None:
        joblib.dump({
            "type": "fpca",
            "n_basis": self.n_basis,
            "var_threshold": self.var_threshold,
            "pca_whiten": self.whiten,
            "frame_pca_model": self.frame_pca.model,
            "score_pca_model": self.score_pca.model,
            "_feature_dim": self._feature_dim,
        }, path)

    @classmethod
    def _from_dict(cls, data: dict) -> "FunctionalPCAExtractor":
        obj = cls(data["frame_pca_model"].n_components, data["n_basis"],
                  data["var_threshold"], data.get("pca_whiten", False))
        obj.frame_pca.model = data["frame_pca_model"]
        obj.frame_pca.fitted = True
        obj.score_pca.model = data["score_pca_model"]
        obj.score_pca.fitted = True
        obj._feature_dim = data["_feature_dim"]
        return obj


class LDAMeanExtractor(FeatureExtractor):
    """Supervised reduction: per-frame PCA → simple temporal mean → LDA coordinates.

    The only label-using extractor (fit requires labels). Linear discriminant analysis on
    the temporally averaged reduced representations yields at most (n_classes - 1)
    discriminant coordinates, on which the probe is then trained.
    """

    def __init__(self, pca_dim: int, whiten: bool):
        from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
        self.whiten = whiten
        self.pca = PCATransformer(n_components=pca_dim, whiten=whiten)
        self.lda = LinearDiscriminantAnalysis()
        self._feature_dim: int | None = None

    def _reduced_mean(self, emb: torch.Tensor) -> torch.Tensor:
        return self.pca.transform(emb.T).mean(dim=0)  # [k]

    def fit(self, embeddings: list[torch.Tensor], labels: list | None = None) -> None:
        if labels is None:
            raise ValueError("LDAMeanExtractor.fit requires training labels.")
        all_frames = torch.cat([e.T for e in embeddings], dim=0)
        self.pca.fit(all_frames)
        means = torch.stack([self._reduced_mean(e) for e in embeddings], dim=0).numpy()
        self.lda.fit(means, list(labels))
        self._feature_dim = self.lda.transform(means[:1]).shape[1]

    def transform(self, emb: torch.Tensor) -> torch.Tensor:
        mean = self._reduced_mean(emb).unsqueeze(0).numpy()
        return torch.from_numpy(self.lda.transform(mean)).float().squeeze(0)

    @property
    def feature_dim(self) -> int:
        if self._feature_dim is None:
            raise RuntimeError("LDAMeanExtractor.feature_dim is only defined after fit().")
        return self._feature_dim

    def save(self, path: Path) -> None:
        joblib.dump({
            "type": "lda_mean",
            "pca_model": self.pca.model,
            "pca_whiten": self.whiten,
            "lda_model": self.lda,
            "_feature_dim": self._feature_dim,
        }, path)

    @classmethod
    def _from_dict(cls, data: dict) -> "LDAMeanExtractor":
        obj = cls(data["pca_model"].n_components, data.get("pca_whiten", False))
        obj.pca.model = data["pca_model"]
        obj.pca.fitted = True
        obj.lda = data["lda_model"]
        obj._feature_dim = data["_feature_dim"]
        return obj


def make_feature_extractor(hp: HyperparametersProbe) -> FeatureExtractor:
    if hp.extractor_type == "path_signature":
        return PathSignatureExtractor(hp.pre_sig_pca_dim, hp.sig_depth, hp.whiten)
    elif hp.extractor_type == "weighted_mean_pca":
        return WeightedMeanPCAExtractor(hp.pca_dim, hp.whiten, hp.decay_rate_end, hp.decay_rate_beg)
    elif hp.extractor_type == "fpca":
        return FunctionalPCAExtractor(hp.pca_dim, hp.fpca_n_basis, hp.fpca_var_threshold, hp.whiten)
    elif hp.extractor_type == "lda_mean":
        return LDAMeanExtractor(hp.pca_dim, hp.whiten)
    elif hp.extractor_type in SEQUENCE_EXTRACTOR_TYPES:
        return PerFramePCAExtractor(hp.pca_dim, hp.whiten)
    raise ValueError(f"Unknown extractor_type: {hp.extractor_type!r}")


_EXTRACTOR_REGISTRY: dict[str, type[FeatureExtractor]] = {
    "weighted_mean_pca": WeightedMeanPCAExtractor,
    "path_signature": PathSignatureExtractor,
    "per_frame_pca": PerFramePCAExtractor,
    "fpca": FunctionalPCAExtractor,
    "lda_mean": LDAMeanExtractor,
}

def load_feature_extractor(path: Path) -> FeatureExtractor:
    data = joblib.load(path)
    extractor_type = data["type"]
    if extractor_type not in _EXTRACTOR_REGISTRY:
        raise ValueError(f"Unknown extractor type in saved file: {extractor_type!r}")
    return _EXTRACTOR_REGISTRY[extractor_type]._from_dict(data)

# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class EmbeddingDataset(Dataset):
    def __init__(self, registry_df: pd.DataFrame, feature_extractor: FeatureExtractor):
        """
        :param registry_df: DataFrame with "path" (to .pt embedding file) and "label" columns.
        :param feature_extractor: A fitted FeatureExtractor instance.
        """
        self.df = registry_df
        self.feature_extractor = feature_extractor
        self.label_to_idx = {label: i for i, label in enumerate(sorted(self.df["label"].unique()))}
        self.idx_to_label = {i: lbl for lbl, i in self.label_to_idx.items()}
        # The extractor is fitted before the dataset is built, so transform(emb) is
        # deterministic; caching it makes epochs after the first read-only (the per-epoch
        # recompute made expensive extractors like FPCA at large pca_dim ~20 min/epoch).
        self._cache: dict[int, tuple] = {}

    def __len__(self):
        return len(self.df)

    def _load_raw(self, idx: int) -> torch.Tensor:
        """Return the raw [D, N] embedding tensor without any transformation."""
        row = self.df.iloc[idx]
        sample = torch.load(row["path"], weights_only=False)
        return sample["embeddings"]

    def __getitem__(self, idx):
        if idx not in self._cache:
            row = self.df.iloc[idx]
            emb = self._load_raw(idx)           # [D, N]
            label = self.label_to_idx[row["label"]]
            feat = self.feature_extractor.transform(emb)    # [feature_dim]
            self._cache[idx] = (feat, label, row["path"])
        return self._cache[idx]

# ---------------------------------------------------------------------------
# Data manager
# ---------------------------------------------------------------------------

class DataManager:
    def __init__(self, dir_probe: Path, feature_extractor: FeatureExtractor, batch_size: int = 32, is_distributed: bool = False, split_tag: str | None = None) -> None:
        self.feature_extractor = feature_extractor
        self.ds: dict[str, EmbeddingDataset] = {}
        samplers: dict[str, DistributedSampler] = {}
        self.dl: dict[str, DataLoader] = {}
        types = ['train', 'dev', 'test']
        suffix = f"_{split_tag}" if split_tag else ""
        effective_batch = math.ceil(batch_size / torch.cuda.device_count()) if is_distributed else batch_size
        collate = pad_collate if feature_extractor.returns_sequence else None
        for t in types:
            self.ds[t] = EmbeddingDataset(pd.read_csv(dir_probe / f"registry_{t}{suffix}.csv"), feature_extractor)
            samplers[t] = DistributedSampler(self.ds[t]) if is_distributed else None
            bs = effective_batch if (t == 'train' and is_distributed) else batch_size
            self.dl[t] = DataLoader(self.ds[t], batch_size=bs, shuffle=(samplers[t] is None), sampler=samplers[t], collate_fn=collate)

    def fit_feature_extractor(self) -> None:
        """Fit the extractor on raw training embeddings (all loaded into memory).

        Labels are passed along for the supervised extractors (LDAMeanExtractor); the
        unsupervised ones ignore them."""
        raw_embs = [self.ds['train']._load_raw(i) for i in range(len(self.ds['train']))]
        labels = self.ds['train'].df["label"].tolist()
        self.feature_extractor.fit(raw_embs, labels=labels)

    def set_feature_extractor(self, extractor: FeatureExtractor) -> None:
        """Replace the extractor (must already be fitted) on all splits."""
        self.feature_extractor = extractor
        for ds in self.ds.values():
            ds.feature_extractor = extractor

    def get_labels(self) -> list[str]:
        return self.ds['train'].label_to_idx.keys()

# ---------------------------------------------------------------------------
# Probe models
# ---------------------------------------------------------------------------

class NonlinearProbe(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, hidden_dim_multiplier: int = 1):
        super().__init__()
        hidden_dim = int(math.ceil(in_dim * hidden_dim_multiplier))
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.LeakyReLU(0.01),
            nn.Linear(hidden_dim, out_dim),
        )
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x):
        return self.net(x)


def _make_head(in_dim: int, out_dim: int, hidden_dim_multiplier: float) -> nn.Module:
    """Classification head shared by the sequence probes: linear, or NonlinearProbe if
    hidden_dim_multiplier > 0 (mirroring the linear/nonlinear convention of ProbeTrainer)."""
    if hidden_dim_multiplier and hidden_dim_multiplier > 0:
        return NonlinearProbe(in_dim, out_dim, hidden_dim_multiplier)
    head = nn.Linear(in_dim, out_dim)
    nn.init.xavier_uniform_(head.weight)
    nn.init.zeros_(head.bias)
    return head


def pad_collate(batch):
    """Collate variable-length [N_i, k] sequence features into a padded batch.

    Returns ((padded [B, N_max, k], mask [B, N_max] bool), labels, paths); the (x, mask)
    pair is what the sequence probes' forward expects, and padding positions are masked
    out of their softmax.
    """
    feats, labels, paths = zip(*batch)
    lengths = [f.shape[0] for f in feats]
    n_max = max(lengths)
    padded = torch.zeros(len(feats), n_max, feats[0].shape[1])
    mask = torch.zeros(len(feats), n_max, dtype=torch.bool)
    for j, (f, n) in enumerate(zip(feats, lengths)):
        padded[j, :n] = f
        mask[j, :n] = True
    return (padded, mask), torch.tensor(labels), list(paths)


class LearnableWeightingProbe(nn.Module):
    """Position-based learnable temporal weighting, trained jointly with the classifier.

    A weighting profile f_theta over normalized position [0,1] is parameterized by n_knots
    learnable values at equally spaced knots, extended by linear interpolation; the frame
    weights are the masked softmax of f_theta(p_i). Initialized to zeros = uniform average.
    """

    def __init__(self, in_dim: int, out_dim: int, n_knots: int = 5,
                 hidden_dim_multiplier: float = 0.0):
        super().__init__()
        self.theta = nn.Parameter(torch.zeros(n_knots))
        self.head = _make_head(in_dim, out_dim, hidden_dim_multiplier)

    def _profile_at(self, positions: torch.Tensor) -> torch.Tensor:
        """Linear interpolation of theta at positions in [0,1] (any shape)."""
        n_knots = self.theta.shape[0]
        x = positions.clamp(0.0, 1.0) * (n_knots - 1)
        lo = x.floor().long().clamp(max=n_knots - 2)
        frac = x - lo
        return self.theta[lo] * (1 - frac) + self.theta[lo + 1] * frac

    def forward(self, x: torch.Tensor, mask: torch.Tensor,
                positions: torch.Tensor | None = None):
        """positions: optional [B, T] normalized positions in [0,1], overriding the default.

        By default position is derived from the mask, as idx / (lengths - 1). That is correct
        while a batch differs only in padding, but it makes the profile *re-stretch* whenever
        frames are dropped: the last retained frame maps to 1.0 and receives the end-of-word
        weight even if it sits a quarter of the way through the utterance. Partial-average
        evaluations (tasks C19/E5) therefore need to supply the positions the retained frames
        had in the full sequence. Passing None reproduces the training-time behaviour exactly.
        """
        # x: [B, T, k]; mask: [B, T] bool (True = real frame)
        if positions is None:
            lengths = mask.sum(dim=1, keepdim=True)                        # [B, 1]
            idx = torch.arange(x.shape[1], device=x.device).unsqueeze(0)   # [1, T]
            positions = idx / (lengths - 1).clamp(min=1)                   # [B, T]
        scores = self._profile_at(positions).masked_fill(~mask, float("-inf"))
        weights = torch.softmax(scores, dim=1)                             # [B, T]
        pooled = (weights.unsqueeze(-1) * x).sum(dim=1)                    # [B, k]
        return self.head(pooled)

    @torch.no_grad()
    def profile(self, n_points: int = 101) -> torch.Tensor:
        """The learned profile f_theta on a uniform grid (for the weighting figures)."""
        grid = torch.linspace(0, 1, n_points, device=self.theta.device)
        return self._profile_at(grid)


class AttentionPoolingProbe(nn.Module):
    """Content-and-position attention pooling with Q independent queries.

    Each of the num_queries learnable queries scores every reduced frame; the frame *keys*
    are the content features concatenated with a fixed sinusoidal encoding of normalized
    position (pe_bands frequencies m*pi*t, m=1..pe_bands, so 2*pe_bands extra key channels),
    which makes the scoring position-aware -- necessary because a purely content-based query
    is permutation-invariant and cannot represent contour shape. The pooled *values* are the
    content features only, so each query yields one content vector of dim k; the Q pooled
    vectors are concatenated and fed to the classification head, which learns how to combine
    them. num_queries=1 with pe_bands=0 recovers plain single-query content-only pooling."""

    def __init__(self, in_dim: int, out_dim: int, num_queries: int = 1, pe_bands: int = 6,
                 hidden_dim_multiplier: float = 0.0):
        super().__init__()
        self.num_queries = num_queries
        self.pe_bands = pe_bands
        key_dim = in_dim + 2 * pe_bands
        self.query = nn.Parameter(torch.empty(num_queries, key_dim))
        nn.init.normal_(self.query, std=key_dim ** -0.5)
        self.head = _make_head(in_dim * num_queries, out_dim, hidden_dim_multiplier)

    def _positional_encoding(self, mask: torch.Tensor,
                             positions: torch.Tensor | None = None) -> torch.Tensor:
        """Fourier features of normalized position t in [0,1] per frame: [B, T, 2*pe_bands].
        Position is length-normalized (idx / (length-1)) so it is invariant to sequence length,
        matching LearnableWeightingProbe; padding rows are irrelevant (masked out downstream).
        `positions` overrides the derived t; see LearnableWeightingProbe.forward for why
        partial-average evaluations must supply it rather than let it be re-derived."""
        if positions is None:
            lengths = mask.sum(dim=1, keepdim=True)                         # [B, 1]
            idx = torch.arange(mask.shape[1], device=mask.device).unsqueeze(0)  # [1, T]
            positions = idx / (lengths - 1).clamp(min=1)                   # [B, T]
        t = positions
        m = torch.arange(1, self.pe_bands + 1, device=mask.device, dtype=t.dtype)
        ang = math.pi * t.unsqueeze(-1) * m                                # [B, T, pe_bands]
        return torch.cat([ang.sin(), ang.cos()], dim=-1)                   # [B, T, 2*pe_bands]

    def forward(self, x: torch.Tensor, mask: torch.Tensor, return_attention: bool = False,
                positions: torch.Tensor | None = None):
        # x: [B, T, k]; mask: [B, T] bool
        keys = (torch.cat([x, self._positional_encoding(mask, positions)], dim=-1)
                if self.pe_bands > 0 else x)
        scores = torch.einsum("btd,qd->btq", keys, self.query)             # [B, T, Q]
        scores = scores.masked_fill(~mask.unsqueeze(-1), float("-inf"))
        attention = torch.softmax(scores, dim=1)                           # [B, T, Q]
        pooled = torch.einsum("btq,btk->bqk", attention, x)                # [B, Q, k]
        pooled = pooled.reshape(pooled.shape[0], -1)                       # [B, Q*k]
        logits = self.head(pooled)
        return (logits, attention) if return_attention else logits

# ---------------------------------------------------------------------------
# Trainer
# ---------------------------------------------------------------------------

class ProbeTrainer:
    def __init__(self, dir_probe: Path, device, hp: HyperparametersProbe,
                 feature_extractor: FeatureExtractor | None = None,
                 nonlinear: bool = False, is_distributed: bool = False,
                 split_tag: str | None = None, run_tag: str | None = None):
        self.dir_probe: Path = dir_probe
        self.device: torch.device = device
        self.hp: HyperparametersProbe = hp
        self.is_distributed: bool = is_distributed
        self.split_tag: str | None = split_tag
        # Distinguishes runs that share probe dir, split and extractor but differ in where
        # their hyperparameters came from (e.g. transferred vs directly tuned on a codebook).
        self.run_tag: str | None = run_tag

        if feature_extractor is not None:
            extractor = feature_extractor   # pre-fitted (e.g. loaded from disk)
        else:
            extractor = make_feature_extractor(hp)

        self.data: DataManager = DataManager(dir_probe, extractor, hp.batch_size, is_distributed, split_tag=split_tag)

        if feature_extractor is None:
            self.data.fit_feature_extractor()

        feature_dim = self.data.feature_extractor.feature_dim
        self.num_classes: int = len(self.data.get_labels())
        self.nonlinear = nonlinear

        # Sequence extractors aggregate inside the model (weights trained jointly with the
        # classifier); all others use a plain linear / nonlinear head on the fixed features.
        if hp.extractor_type == "learnable_weighting":
            self.model = LearnableWeightingProbe(
                feature_dim, self.num_classes, n_knots=hp.weighting_knots,
                hidden_dim_multiplier=(hp.hidden_dim_multiplier if nonlinear else 0.0),
            ).to(self.device)
        elif hp.extractor_type == "attention_pooling":
            self.model = AttentionPoolingProbe(
                feature_dim, self.num_classes,
                num_queries=hp.num_queries, pe_bands=hp.pe_bands,
                hidden_dim_multiplier=(hp.hidden_dim_multiplier if nonlinear else 0.0),
            ).to(self.device)
        elif nonlinear:
            self.model = NonlinearProbe(
                in_dim=feature_dim,
                out_dim=self.num_classes,
                hidden_dim_multiplier=self.hp.hidden_dim_multiplier,
            ).to(self.device)
        else:
            self.model = nn.Linear(feature_dim, self.num_classes).to(self.device)

        if is_distributed:
            self.model = DDP(self.model, device_ids=[self.device.index], output_device=self.device.index)

        # Per-sample loss weighting (hp.class_weighting). Off by default, so every run
        # trained before 2026-08-13 is reproduced bit-for-bit by the "none" branch.
        #
        # "balanced"          equalize the classes: w propto 1 / count(class).
        # "balanced_by_tune"  equalize the classes AND, inside a class that merges several
        #                     tunes, equalize those tunes:
        #                         w propto 1 / (n_tunes_in_class * count(tune)) .
        #                     A merged class such as lhh-llh otherwise lets its more frequent
        #                     constituent dominate the class it stands for. This cannot be
        #                     expressed as CrossEntropyLoss(weight=...), which weights by
        #                     target *class*, so we weight per sample instead, keyed on the
        #                     path that the collate already carries through every batch.
        self._sample_weight: dict[str, float] | None = None
        if hp.class_weighting != "none":
            self._sample_weight = self._build_sample_weights(hp.class_weighting)
            self.criterion = nn.CrossEntropyLoss(reduction="none")
        else:
            self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.hp.learning_rate,
                                    weight_decay=(1e-5 if nonlinear else 0))
        self.optimizer_late = torch.optim.SGD(self.model.parameters(), lr=1e-4, momentum=0.9, weight_decay=1e-5)

    def train_and_report(self, test: bool = False, backup: bool = False, save_predictions: bool = False):
        reporting_list, acc = self._train(test)
        if test:
            test_acc = self._compute_accuracy(self.data.dl['test'])
        # device.index is 0 for cuda:0 and for DDP rank 0, but None for a plain "cpu" device;
        # single-device CPU training must still save (only DDP rank>0 should skip).
        if self.device.index is None or self.device.index == 0:
            if test:
                print(f"Test accuracy: {test_acc:.3f}")

            if backup or save_predictions:
                timestamp = get_timestamp()
                tag_part = f"-{self.split_tag}" if self.split_tag else ""
                ext_part = ("" if self.hp.extractor_type == "weighted_mean_pca"
                            else f"-{self.hp.extractor_type}")
                nl_part = "-nonlinear" if self.nonlinear else ""
                run_part = f"-{self.run_tag}" if self.run_tag else ""
                out_dir = (self.dir_probe /
                           f"run_{timestamp}{tag_part}{ext_part}{nl_part}{run_part}")
                out_dir.mkdir(parents=True, exist_ok=True)

            if backup:
                print("Saving probe and metadata, timestamp:", timestamp)
                self._save_run_details(out_dir, reporting_list)
                print(f"Saved model, extractor, and metadata to {out_dir}")

            if save_predictions:
                out_path = (out_dir / "predictions_best_probe.csv")
                self._save_model_predictions_original_data(out_path)
                print(f"Saved predictions on full dataset to {str(out_path)}")
        return acc

    def train_optuna(self, trial):
        optimizer = self.optimizer
        switched_optimizer = False
        for epoch in range(self.hp.n_epochs):
            total_loss = self._train_epoch(epoch, optimizer)
            total_loss = torch.tensor([total_loss], device=self.device)
            num_samples = float(len(self.data.dl['train'].dataset))
            if self.is_distributed:
                dist.all_reduce(total_loss, op=dist.ReduceOp.SUM)
            avg_loss = total_loss.item() / num_samples
            acc = self._selection_metric(self.data.dl['dev'])

            if (not switched_optimizer) and avg_loss < 1e-4:
                print(f"Switching to SGD at epoch {epoch}, loss={avg_loss:.7f}")
                switched_optimizer = True
                optimizer = self.optimizer_late

            trial.report(acc, epoch)
            if trial.should_prune():
                raise optuna.exceptions.TrialPruned(f"Trial was pruned at epoch {epoch}.")
        return acc

    def _save_run_details(self, out_dir: Path, reporting_list):
        if len(reporting_list) > 0:
            report_df = pd.DataFrame(reporting_list)
            report_df.to_csv(out_dir / "training_report.csv", index=False)

        torch.save(self.model.state_dict(), out_dir / "best_probe.pt")
        self.data.feature_extractor.save(out_dir / "feature_extractor.joblib")
        save_dataclass_to_json(self.hp, out_dir / "hyperparams.json")

    def _batch_to_device(self, xb):
        """Move a feature batch to the device: a plain [B, q] tensor, or the (padded, mask)
        pair produced by pad_collate for sequence extractors."""
        if isinstance(xb, (tuple, list)):
            x, mask = xb
            return (x.to(self.device, dtype=torch.float32), mask.to(self.device))
        return xb.to(self.device, dtype=torch.float32)

    def _build_sample_weights(self, scheme: str) -> dict[str, float]:
        """path -> loss weight, from the *training* registry only.

        The registry carries both `tune` (the raw eight-way label) and `label` (the possibly
        merged class), which is what makes the by-tune scheme expressible at all. Weights are
        normalized to mean 1 over the training split so that the loss keeps its scale and
        learning rates transfer from the unweighted studies.
        """
        df = self.data.ds['train'].df
        if scheme not in {"balanced", "balanced_by_tune"}:
            raise ValueError(f"unknown class_weighting {scheme!r}")
        if scheme == "balanced_by_tune" and "tune" not in df.columns:
            raise ValueError("class_weighting='balanced_by_tune' needs a 'tune' column in the "
                             "registry; rebuild the splits with buil-all-data-splits.py")
        class_count = df["label"].value_counts().to_dict()
        if scheme == "balanced":
            raw = {p: 1.0 / class_count[c] for p, c in zip(df["path"], df["label"])}
        else:
            tune_count = df["tune"].value_counts().to_dict()
            n_tunes = df.groupby("label")["tune"].nunique().to_dict()
            raw = {p: 1.0 / (n_tunes[c] * tune_count[t])
                   for p, c, t in zip(df["path"], df["label"], df["tune"])}
        mean_w = sum(raw.values()) / len(raw)
        return {p: w / mean_w for p, w in raw.items()}

    def _prefix_truncate(self, xb):
        """Randomly truncate each sequence to a prefix, keeping ORIGINAL frame positions.

        Data augmentation for the partial-input probes: the probe sees the opening of an
        utterance but is still asked for the label of the whole one. Positions are passed
        explicitly rather than left to be derived from the mask, so a retained frame keeps the
        position it held in the full sequence; without this the probe's position axis
        re-stretches over the prefix and the last retained frame collects the end-of-sequence
        weight wherever it happens to sit.
        """
        padded, mask = xb
        lengths = mask.sum(dim=1)                                          # [B]
        idx = torch.arange(padded.shape[1], device=padded.device).unsqueeze(0)
        positions = idx / (lengths - 1).clamp(min=1).unsqueeze(1)          # full-sequence axis
        keep = torch.rand(padded.shape[0], device=padded.device) < self.hp.train_prefix_augment
        frac = (self.hp.prefix_augment_min
                + (1.0 - self.hp.prefix_augment_min)
                * torch.rand(padded.shape[0], device=padded.device))
        n_keep = torch.where(keep, (frac * lengths).ceil().long().clamp(min=1), lengths)
        new_mask = mask & (idx < n_keep.unsqueeze(1))
        return (padded, new_mask, positions)

    def _forward(self, xb):
        """Forward pass; xb must already be on the device (see _batch_to_device)."""
        if isinstance(xb, tuple):
            if len(xb) == 3:
                # (padded, mask, positions) from _prefix_truncate. Passed by keyword because
                # AttentionPoolingProbe.forward takes `return_attention` third positionally.
                return self.model(xb[0], xb[1], positions=xb[2])
            return self.model(*xb)
        return self.model(xb)

    def _compute_balanced_accuracy(self, dl: DataLoader) -> torch.Tensor:
        """Mean per-class recall: chance is 1/n_classes whatever the class proportions.

        Used as the model-selection metric whenever the loss is class-weighted. Selecting on
        raw accuracy while training on a balanced loss makes the two disagree: on a problem
        whose majority class holds 0.70 of the data, a trial that collapses onto that class
        scores 0.70 and survives pruning, which is exactly the behaviour the weighted loss is
        there to prevent.
        """
        self.model.eval()
        n_cls = self.num_classes
        correct = torch.zeros(n_cls, device=self.device)
        total = torch.zeros(n_cls, device=self.device)
        with torch.no_grad():
            for xb, yb, _ in dl:
                xb, yb = self._batch_to_device(xb), yb.to(self.device)
                preds = self._forward(xb).argmax(dim=1)
                for c in range(n_cls):
                    m = yb == c
                    correct[c] += (preds[m] == c).sum()
                    total[c] += m.sum()
        if self.is_distributed:
            dist.all_reduce(correct, op=dist.ReduceOp.SUM)
            dist.all_reduce(total, op=dist.ReduceOp.SUM)
        present = total > 0
        return (correct[present] / total[present]).mean()

    def _selection_metric(self, dl: DataLoader) -> torch.Tensor:
        """The quantity HPO optimizes: balanced accuracy iff the loss is class-weighted."""
        if self._sample_weight is not None:
            return self._compute_balanced_accuracy(dl)
        return self._compute_accuracy(dl)

    def _compute_accuracy(self, dl: DataLoader) -> torch.Tensor:
        self.model.eval()
        correct, total = 0.0, 0.0
        with torch.no_grad():
            for xb, yb, _ in dl:
                xb, yb = self._batch_to_device(xb), yb.to(self.device)
                preds = self._forward(xb).argmax(dim=1)
                correct += (preds == yb).sum()
                total += torch.tensor(yb.size(0)).to(self.device)
        if self.is_distributed:
            dist.all_reduce(correct, op=dist.ReduceOp.SUM)
            dist.all_reduce(total, op=dist.ReduceOp.SUM)
        return correct / total

    def _train(self, test: bool = None):
        reporting_list = []
        optimizer = self.optimizer
        switch_to_sgd = False
        for epoch in range(self.hp.n_epochs):
            total_loss = self._train_epoch(epoch, optimizer)
            total_loss = torch.tensor([total_loss], device=self.device)
            num_samples = float(len(self.data.dl['train'].dataset))
            if self.is_distributed:
                dist.all_reduce(total_loss, op=dist.ReduceOp.SUM)
            avg_loss = total_loss.item() / num_samples
            acc = self._compute_accuracy(self.data.dl['dev'])
            if (not switch_to_sgd) and avg_loss < 1e-4:
                print(f"Switching to SGD at epoch {epoch}, loss={avg_loss:.7f}")
                switch_to_sgd = True
                optimizer = self.optimizer_late
            if test:
                reporting_list.append({"epoch": epoch + 1, "avg_loss": avg_loss, "dev_acc": acc})
                print(f"Epoch {epoch+1:02d}: loss={avg_loss:.7f}, dev_acc={acc:.3f}")
        return reporting_list, acc

    def _train_epoch(self, index: int, optimizer):
        if self.is_distributed:
            self.data.dl['train'].sampler.set_epoch(index)
        self.model.train()
        total_loss = 0.0
        for xb, yb, paths in self.data.dl['train']:
            xb, yb = self._batch_to_device(xb), yb.to(self.device)
            if self.hp.train_prefix_augment > 0.0:
                if not isinstance(xb, tuple):
                    raise ValueError("train_prefix_augment needs a sequence extractor "
                                     f"(got extractor_type={self.hp.extractor_type!r})")
                xb = self._prefix_truncate(xb)
            optimizer.zero_grad()
            logits = self._forward(xb)
            loss = self.criterion(logits, yb)
            if self._sample_weight is not None:
                # criterion has reduction="none" in this branch: [B] per-sample losses.
                wb = torch.tensor([self._sample_weight[p] for p in paths],
                                  device=self.device, dtype=loss.dtype)
                loss = (loss * wb).sum() / wb.sum()
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * yb.size(0)
        return total_loss

    def _save_model_predictions_original_data(self, out_path: Path, show_probas: bool = True):
        loaders = [self.data.dl['train'], self.data.dl['dev'], self.data.dl['test']]
        names = ["train", "dev", "test"]
        preds_df = self._compute_predictions_df(loaders, names, show_probas)
        preds_df["correct"] = (preds_df['label'] == preds_df['prediction']).astype('int')
        preds_df.to_csv(out_path, index=False)

    def get_model_predictions_new_data(self, data_registry_df: pd.DataFrame, include_probas: bool = False):
        ds = EmbeddingDataset(data_registry_df, feature_extractor=self.data.feature_extractor)
        collate = pad_collate if self.data.feature_extractor.returns_sequence else None
        dl = DataLoader(ds, batch_size=32, collate_fn=collate)
        preds_df = self._compute_predictions_df([dl], ["unseen"], include_probas)
        return preds_df

    def _compute_predictions_df(self, loaders: list[DataLoader], names: list[str], show_probas: bool = False):
        self.model.eval()
        preds_list = []
        with torch.no_grad():
            for loader, split_name in zip(loaders, names):
                ds: EmbeddingDataset = loader.dataset
                for embs, labels, paths in loader:
                    embs = self._batch_to_device(embs)
                    labels = labels.to(self.device)
                    logits = self._forward(embs)
                    probabilities = torch.softmax(logits, dim=1)
                    preds = logits.argmax(dim=1).cpu().numpy()
                    for p, lab, pred, prob in zip(paths, labels, preds, probabilities):
                        new_entry = {
                            "path": p,
                            "label": ds.idx_to_label[int(lab)],
                            "prediction": self.data.ds['train'].idx_to_label[int(pred)],
                        }
                        if show_probas:
                            new_entry["probabilities"] = [f"{self.data.ds['train'].idx_to_label[i]}: {p}" for i, p in enumerate(prob)]
                        new_entry["data"] = split_name
                        preds_list.append(new_entry)
        return pd.DataFrame(preds_list)

    @classmethod
    def load_trained_probe(cls, device: torch.device, probe_dir: Path, run_dir: Path):
        hp = load_dataclass_from_json(HyperparametersProbe, run_dir / "hyperparams.json")

        extractor_path = run_dir / "feature_extractor.joblib"
        if extractor_path.exists():
            extractor = load_feature_extractor(extractor_path)
        else:
            # Legacy: reconstruct WeightedMeanPCAExtractor from old pca_transform.joblib
            pca_path = run_dir / "pca_transform.joblib"
            extractor = WeightedMeanPCAExtractor(hp.pca_dim, hp.whiten, hp.decay_rate_end, hp.decay_rate_beg)
            extractor.pca = PCATransformer.load(pca_path)

        nonlinear = (hp.hidden_dim_multiplier != 0.0)
        probe = cls(probe_dir, device, hp, feature_extractor=extractor, nonlinear=nonlinear)
        probe.model.to(device)

        state_dict = torch.load(run_dir / "best_probe.pt", map_location=device)
        state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
        probe.model.load_state_dict(state_dict)
        probe.model.eval()
        return probe


def save_dataclass_to_json(dataclass_obj, path):
    with open(path, "w") as f:
        json.dump(dataclasses.asdict(dataclass_obj), f, indent=4)
    print(f"Saved object of type {type(dataclass_obj)} to {path}")

def load_dataclass_from_json(cls, path: Path):
    print(f"Loading object of type {str(cls)} at {path}")
    with open(path, "r") as f:
        data = json.load(f)
    # Keep only known fields so old JSON with extra keys and new dataclass with added-default
    # fields both load cleanly (missing keys use their declared defaults).
    known = {f.name for f in dataclasses.fields(cls)}
    return cls(**{k: v for k, v in data.items() if k in known})
