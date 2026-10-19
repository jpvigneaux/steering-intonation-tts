#!/usr/bin/env python3
"""Is the tune linearly decodable from the backbone state the level-0 head reads?

The paper's result is that the tune is *expressed* at quantizer levels 1 and 2, which only the
depth model emits, and not at level 0. That is a statement about the codec's levels, and on its
own it says nothing about the backbone: the whole stack is generated from ``h_t``, so whatever
fixes the tune has to be present there. This asks the question directly.

Three things are measured on the same recordings, with the same classifier and the same folds,
which is what makes them comparable:

* **the backbone state.** ``h_t`` pooled over several regions of the utterance -- the whole
  recording, the nuclear word, its final fifth, its first fifth, the pre-nuclear region, and the
  single final frame. Comparing regions localizes *when* the backbone state carries the tune,
  which the codebook measurements cannot see.
* **the codeword vectors.** The quantity in the paper's depth profile, evaluated on generated
  rather than encoded material: level k's chosen codeword vectors, pooled over the same regions.
  This is Family D of the read-out families.
* **a permutation null.** Labels shuffled within the cross-validation, so the chance level is
  measured on this sample rather than assumed to be 0.5.

Folds are **leave-one-carrier-sentence-out**. Six carrier sentences give six folds, and a probe
is never tested on a sentence it was trained on, so a probe that has merely memorized which
words go with which tune cannot score. That is stricter than a random split and is the reason
the numbers here are lower than a random-split probe would report.

Output: ``backbone_probe.csv`` and a printed summary.

With ``--save-directions`` it also writes the fitted directions themselves, which is what the
backbone-steering intervention consumes. One direction per held-out carrier sentence, fitted on
the other five, so a recording is never steered along a direction fitted with it in the training
set --- the same discipline the accuracy above is measured under, applied to the vector rather
than to the score.
"""
from __future__ import annotations

import argparse
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT / "src"))

from interventions import alignment as AL, archive as A, config  # noqa: E402

warnings.filterwarnings("ignore")

EDGE_FRACTION = 0.2


# ----------------------------------------------------------------- feature construction
def regions(n_frames: int, nuclear: slice | None) -> dict[str, np.ndarray]:
    """Frame index sets, named. An empty region is dropped by the caller, not imputed."""
    everything = np.arange(n_frames)
    out = {"all": everything, "final": everything[-1:]}
    if nuclear is None:
        return out
    span = np.arange(nuclear.start, min(nuclear.stop, n_frames))
    if span.size == 0:
        return out
    width = max(1, int(round(span.size * EDGE_FRACTION)))
    out["nuclear"] = span
    out["offset"] = span[-width:]
    out["onset"] = span[:width]
    pre = everything[:nuclear.start]
    if pre.size:
        out["prenuclear"] = pre
    return out


def codeword_vectors(tokens: np.ndarray, codebooks: dict[int, np.ndarray],
                     level: int) -> np.ndarray | None:
    """Level ``level``'s chosen codeword vectors, ``(T, D)`` -- the paper's zeta_k."""
    table = codebooks.get(level)
    if table is None:
        return None
    ids = np.clip(tokens[:, level], 0, table.shape[0] - 1)
    return table[ids]


def load_codebooks(levels: list[int], device: str = "cpu") -> dict[int, np.ndarray]:
    """Mimi's per-level codebook matrices, so token ids become the vectors of eq. (3).

    Mimi splits its stack: ``rvq_first`` holds the semantic level 0 and ``rvq_rest`` the
    acoustic levels above it. Walking them in order recovers the level numbering the archives
    use.
    """
    import torch
    from basic8probes.audio import MimiAudioManager

    manager = MimiAudioManager(device, num_codebooks=max(levels) + 1)
    quantizer = manager.mimi.quantizer
    layers = list(quantizer.rvq_first.vq.layers) + list(quantizer.rvq_rest.vq.layers)

    tables: dict[int, np.ndarray] = {}
    for level in levels:
        if level >= len(layers):
            continue
        with torch.no_grad():
            tables[level] = layers[level]._codebook.embedding.detach().cpu().numpy()
    return tables


# ----------------------------------------------------------------- evaluation
def leave_one_sentence_out(features: np.ndarray, labels: np.ndarray,
                           groups: np.ndarray, *, seed: int = 0) -> dict:
    """Accuracy and AUC under leave-one-carrier-sentence-out, with a permutation null."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    def run(y: np.ndarray) -> tuple[float, float]:
        scores = np.zeros(len(y), dtype=float)
        for held in np.unique(groups):
            test = groups == held
            train = ~test
            if len(np.unique(y[train])) < 2:
                scores[test] = 0.5
                continue
            model = make_pipeline(
                StandardScaler(),
                LogisticRegression(max_iter=5000, C=1.0, solver="lbfgs"))
            model.fit(features[train], y[train])
            scores[test] = model.predict_proba(features[test])[:, 1]
        accuracy = float(np.mean((scores > 0.5).astype(int) == y))
        auc = float(roc_auc_score(y, scores)) if len(np.unique(y)) > 1 else float("nan")
        return accuracy, auc

    accuracy, auc = run(labels)

    rng = np.random.default_rng(seed)
    null = []
    for _ in range(20):
        shuffled = labels.copy()
        # Permute within each carrier sentence, so the null keeps the design's balance.
        for held in np.unique(groups):
            index = np.where(groups == held)[0]
            shuffled[index] = rng.permutation(shuffled[index])
        null.append(run(shuffled)[0])

    return {"accuracy": accuracy, "auc": auc,
            "null_accuracy_mean": float(np.mean(null)),
            "null_accuracy_max": float(np.max(null)),
            "n": int(len(labels)), "d": int(features.shape[1])}


def fold_directions(features: np.ndarray, labels: np.ndarray,
                    groups: np.ndarray) -> tuple[np.ndarray, np.ndarray, list[dict]]:
    """One steering direction per held-out carrier sentence, fitted without it.

    The probe standardizes its inputs, so its coefficients live in the standardized space. A
    direction to be added to a raw backbone state has to live in the raw space, which is what
    dividing by the scaler's per-feature standard deviation gives: the linear functional the
    probe computes is unchanged, expressed in the coordinates ``h_t`` actually uses.

    Each direction is returned as a unit vector oriented so that the interrogative condition
    lies in the positive direction, together with ``gap`` --- the distance between the two
    conditions' training means projected on it. ``gap`` is the natural unit for a steering
    coefficient: adding ``1 x gap`` moves a state by exactly the separation the conditions show
    along that direction, and unlike the raw coefficient norm it is comparable across folds.
    """
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    held_out = np.unique(groups)
    vectors, gaps, rows = [], [], []
    for held in held_out:
        test = groups == held
        train = ~test
        model = make_pipeline(StandardScaler(),
                              LogisticRegression(max_iter=5000, C=1.0, solver="lbfgs"))
        model.fit(features[train], labels[train])
        scaler = model.named_steps["standardscaler"]
        classifier = model.named_steps["logisticregression"]

        raw = classifier.coef_[0] / scaler.scale_
        norm = float(np.linalg.norm(raw))
        unit = raw / norm
        projection = features @ unit
        mean_a = float(projection[train & (labels == 0)].mean())
        mean_i = float(projection[train & (labels == 1)].mean())
        if mean_i < mean_a:
            # Orientation is a convention of the label encoding, not a result. Fixing it here
            # means a positive steering coefficient always means "toward the interrogative".
            unit, projection = -unit, -projection
            mean_a, mean_i = -mean_a, -mean_i

        held_scores = model.predict_proba(features[test])[:, 1]
        held_labels = labels[test]
        vectors.append(unit.astype(np.float32))
        gaps.append(mean_i - mean_a)
        rows.append({
            "held_out": str(held),
            "n_train": int(train.sum()), "n_held_out": int(test.sum()),
            "gap": mean_i - mean_a,
            "mean_projection_A": mean_a, "mean_projection_I": mean_i,
            "within_sd": float(projection[train].std(ddof=1)),
            "coefficient_norm": norm,
            "held_out_accuracy": float(np.mean((held_scores > 0.5).astype(int) == held_labels)),
            "held_out_auc": (float(roc_auc_score(held_labels, held_scores))
                             if len(np.unique(held_labels)) > 1 else float("nan")),
        })
    return np.stack(vectors), np.asarray(gaps, dtype=np.float32), rows


def difference_in_means_directions(frames: list[np.ndarray], labels: np.ndarray,
                                   groups: np.ndarray) -> tuple[np.ndarray, np.ndarray, list[dict]]:
    """Steering directions from the difference between the conditions' mean states.

    A discriminative probe and a difference of means answer different questions, and only the
    second is a statement about *where the conditions sit*. Logistic regression returns the
    direction that separates them best, which is free to point along a low-variance nuisance
    axis if doing so sharpens the boundary; adding that direction to a state moves it somewhere
    the data never goes. The difference of the class means points from one condition's centre to
    the other's by construction, so an edit of length ``gap`` lands a frame on the other
    condition's mean.
    That is the edit a steering method actually wants, and it is why this is worth measuring
    beside the probe rather than instead of it.

    ``frames`` holds one ``(T_i, D)`` array per recording --- every frame of the pooling region,
    not the recording's pooled mean --- so the means are taken over token positions, pooled over
    recordings. Folds are leave-one-carrier-sentence-out, as for the probe, so no recording is
    steered along a direction its own sentence helped fit.

    ``gap`` is the norm of the difference, which is exactly what the two means project to on the
    unit direction; it therefore plays the same role as the probe's gap and the coefficient
    ``alpha`` means the same thing on both scales.
    """
    held_out = np.unique(groups)
    vectors, gaps, rows = [], [], []
    for held in held_out:
        train = groups != held
        declarative = np.concatenate([f for f, t, y in zip(frames, train, labels) if t and y == 0])
        interrogative = np.concatenate([f for f, t, y in zip(frames, train, labels) if t and y == 1])
        delta = interrogative.mean(axis=0) - declarative.mean(axis=0)
        norm = float(np.linalg.norm(delta))
        if norm == 0.0:
            raise ValueError(f"the two conditions have the same mean state on fold {held!r}")
        unit = delta / norm

        # A mass-mean classifier on the same direction, so its held-out accuracy is comparable
        # with the probe's: score a recording by its pooled projection against the midpoint of
        # the training means.
        pooled = np.stack([f.mean(axis=0) for f in frames])
        projection = pooled @ unit
        midpoint = 0.5 * ((interrogative.mean(axis=0) + declarative.mean(axis=0)) @ unit)
        predicted = (projection > midpoint).astype(int)
        test = ~train

        vectors.append(unit.astype(np.float32))
        gaps.append(norm)
        rows.append({
            "held_out": str(held), "method": "diffmeans",
            "n_train": int(train.sum()), "n_held_out": int(test.sum()),
            "n_train_frames": int(declarative.shape[0] + interrogative.shape[0]),
            "gap": norm,
            "mean_projection_A": float(declarative.mean(axis=0) @ unit),
            "mean_projection_I": float(interrogative.mean(axis=0) @ unit),
            "within_sd": float(projection[train].std(ddof=1)),
            "coefficient_norm": norm,
            "held_out_accuracy": float(np.mean(predicted[test] == labels[test])),
            "held_out_auc": float("nan"),
        })
    return np.stack(vectors), np.asarray(gaps, dtype=np.float32), rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--archives", default=None)
    parser.add_argument("--audio-root", default=None)
    parser.add_argument("--levels", type=int, nargs="+", default=[0, 1, 2, 3, 7],
                        help="Quantizer levels to probe alongside the backbone state.")
    parser.add_argument("--no-codebooks", action="store_true",
                        help="Skip the codeword read-outs; probe only the backbone state.")
    parser.add_argument("--save-directions", action="store_true",
                        help="Also write the per-fold steering directions, which "
                             "pipeline/patch_backbone.py consumes.")
    parser.add_argument("--direction-region", default="nuclear",
                        help="Which pooling region the saved directions are fitted on.")
    parser.add_argument("-o", "--out", default=None)
    args = parser.parse_args()

    archives_root = Path(args.archives) if args.archives else config.archives_root()
    audio_root = Path(args.audio_root) if args.audio_root else archives_root / "outputs"

    records = []
    for path in sorted((archives_root / A.LATENTS_DIR).glob("*.npz")):
        archive = A.load(path, with_internals=True)
        if archive.last_h is None:
            continue
        try:
            nuclear = AL.nuclear_window(archive, AL.textgrid_for(archive, audio_root))
        except (FileNotFoundError, ValueError):
            nuclear = None
        records.append((archive, nuclear))

    print(f"{len(records)} archives with a backbone state, "
          f"{sum(n is not None for _, n in records)} with an alignment")
    if not records:
        raise SystemExit("no archives carry last_h; regenerate with the instrumented loop")

    labels = np.array([1 if a.condition == "I" else 0 for a, _ in records])
    groups = np.array([a.name for a, _ in records])
    print(f"labels: {int((labels == 0).sum())} A, {int((labels == 1).sum())} I; "
          f"{len(np.unique(groups))} carrier sentences")

    codebooks: dict[int, np.ndarray] = {}
    if not args.no_codebooks:
        try:
            codebooks = load_codebooks(args.levels)
            print(f"codebooks loaded for levels {sorted(codebooks)} "
                  f"(dim {next(iter(codebooks.values())).shape[1]})")
        except Exception as exc:                                    # noqa: BLE001
            print(f"codebooks unavailable ({exc}); probing the backbone state only")

    # One matrix per (representation, region). A recording missing a region is dropped from
    # that row only, and the row reports its own n.
    banks: dict[str, dict[str, np.ndarray]] = {}
    for index, (archive, nuclear) in enumerate(records):
        pools = regions(archive.n_frames, nuclear)
        sources = {"h": archive.last_h}
        for level in sorted(codebooks):
            vectors = codeword_vectors(archive.tokens, codebooks, level)
            if vectors is not None:
                sources[f"c{level}"] = vectors
        for source_name, matrix in sources.items():
            for region_name, frames in pools.items():
                frames = frames[frames < matrix.shape[0]]
                if frames.size == 0:
                    continue
                key = f"{source_name}:{region_name}"
                banks.setdefault(key, {})[index] = matrix[frames].mean(axis=0)

    rows = []
    for key, byindex in sorted(banks.items()):
        index = np.array(sorted(byindex))
        if len(index) < len(records) * 0.8:
            print(f"  {key}: only {len(index)} of {len(records)} recordings; skipped")
            continue
        features = np.stack([byindex[i] for i in index])
        result = leave_one_sentence_out(features, labels[index], groups[index])
        rows.append({"representation": key.split(":")[0], "region": key.split(":")[1],
                     **result})

    table = pd.DataFrame(rows).sort_values(["representation", "region"])
    out = Path(args.out) if args.out else Path(config.get("analysis_root"))
    out.mkdir(parents=True, exist_ok=True)
    table.to_csv(out / "backbone_probe.csv", index=False)

    if args.save_directions:
        key = f"h:{args.direction_region}"
        if key not in banks:
            raise SystemExit(f"no {key} features; regions available: {sorted(banks)}")
        index = np.array(sorted(banks[key]))
        features = np.stack([banks[key][i] for i in index])
        vectors, gaps, fold_rows = fold_directions(features, labels[index], groups[index])
        for row in fold_rows:
            row["method"] = "probe"

        # The two directions answer different questions and are saved side by side, so the
        # steering sweep can be run on either without refitting: the probe's separates the
        # conditions, the difference of means points from one to the other.
        frames = [np.asarray(records[i][0].last_h)[
            regions(records[i][0].n_frames, records[i][1])[args.direction_region]]
            for i in index]
        mean_vectors, mean_gaps, mean_rows = difference_in_means_directions(
            frames, labels[index], groups[index])

        payload = {
            "sentences": np.array([row["held_out"] for row in fold_rows]),
            "directions": vectors, "gaps": gaps,               # the default, kept for callers
            "directions_probe": vectors, "gaps_probe": gaps,
            "directions_diffmeans": mean_vectors, "gaps_diffmeans": mean_gaps,
            "region": np.array(args.direction_region), "d": np.array(features.shape[1]),
        }
        np.savez_compressed(out / "backbone_directions.npz", **payload)
        folds = pd.DataFrame(fold_rows + mean_rows)
        folds.to_csv(out / "backbone_directions.csv", index=False)

        cosines = [float(v @ m) for v, m in zip(vectors, mean_vectors)]
        print(f"\n=== steering directions on {key} "
              f"({vectors.shape[0]} folds, d={vectors.shape[1]}) ===")
        print(folds.to_string(index=False, float_format=lambda v: f"{v:.3f}"))
        print(f"cosine(probe, difference of means) per fold: "
              f"{', '.join(f'{c:.3f}' for c in cosines)}")
        print(f"Wrote {out / 'backbone_directions.npz'} and backbone_directions.csv")

    print("\n=== linear decodability, leave-one-carrier-sentence-out ===")
    print(table.to_string(index=False, float_format=lambda v: f"{v:.3f}"))
    print(f"\nWrote {out / 'backbone_probe.csv'}")


if __name__ == "__main__":
    main()
