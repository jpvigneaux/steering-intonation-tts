#!/usr/bin/env python3
"""Does the contrast live in the target word and what follows it, and nowhere before?

Each recording is split at the start of its designed target word, located in the forced
alignment's ``words`` tier **by its label**, not by position -- the target need not be final,
and whether it was is recorded rather than assumed:

    x_pre  = x restricted to [0, s)          x_post = x restricted to [s, end]

with ``s`` the target word's start. Each segment is represented by its F0 contour in ERB over
its voiced extent, resampled to ``N`` points (``f0.f0_trajectory`` with an explicit interval),
under both centering schemes, and projected on the principal components explaining 90% of its
variance (see :func:`principal_scores` for why). Then, separately for the bag of pre segments
and the bag of post segments, the number of clusters is selected over k = 1..K.

The prediction: the post bag clusters into two groups that coincide with the generating
condition, and the pre bag does not. That is a statement about k = 1 as much as about k = 2, and
of the criteria only two are defined at k = 1: the gap statistic and the Gaussian-mixture BIC.
Calinski-Harabasz, silhouette and Davies-Bouldin need two clusters to exist and so can never
conclude that there is no structure. The gap is read with Tibshirani's rule -- the smallest k
with gap(k) >= gap(k+1) - s(k+1) -- rather than its argmax, which is what lets it choose k = 1.

**Confound, and its control.** Material before the target differs between carrier sentences
(``He modeled`` against ``She remained with``), so a bag pooling sentences would cluster by
sentence. Either keep sentences apart (``--bag-by``), or subtract each sentence's mean contour
within the bag (``--center-within``), which removes the lexical shape and keeps the
within-sentence contrast provided both conditions occur in every sentence.

Outputs ``split_clustering.csv`` (every criterion at every k) and ``split_clustering_summary.csv``
(the k each criterion selects, and the agreement of the k = 2 partition with the labels).
"""
from __future__ import annotations

import argparse
import re
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from praatio import textgrid as tgio

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT / "src"))

from gap_selection import evaluate, principal_scores, selected  # noqa: E402
from interventions import config, f0 as F  # noqa: E402

warnings.filterwarnings("ignore", category=RuntimeWarning)

SILENCE = {"", "sp", "sil", "sp1", "<eps>", "spn"}


def target_interval(textgrid: Path, word: str) -> tuple[float, float, bool]:
    """The last interval labelled ``word`` (case-insensitive), and whether it is utterance-final.

    Final means no non-silent word follows it.
    """
    grid = tgio.openTextgrid(str(textgrid), includeEmptyIntervals=True)
    entries = [e for e in grid.getTier("words").entries if e.label.strip().lower() not in SILENCE]
    matches = [i for i, e in enumerate(entries) if e.label.strip().lower() == word.lower()]
    if not matches:
        raise ValueError(f"no word {word!r} in {textgrid.name}")
    i = matches[-1]
    return float(entries[i].start), float(entries[i].end), i == len(entries) - 1


def segments(wav: Path, textgrid: Path, word: str, n_points: int) -> dict:
    start, _, final = target_interval(textgrid, word)
    begin, end = F.voiced_extent(wav)
    if not begin < start < end:
        raise ValueError(f"target start {start:.3f}s outside the voiced extent of {wav.name}")
    pre = F.f0_trajectory(wav, interval=(begin, start), n_points=n_points)
    post = F.f0_trajectory(wav, interval=(start, end), n_points=n_points)
    return {"pre": pre.f0_erb, "post": post.f0_erb, "split_s": start, "target_is_final": final}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--audio", nargs="+", required=True)
    parser.add_argument("--stem-regex", required=True,
                        help="Named groups: 'word' (the target word) and 'label' (the generating "
                             "condition or tune) are required; any others become columns.")
    parser.add_argument("--keep", nargs="*", default=None,
                        help="FIELD=V1,V2 filters, e.g. label=hll,lhh speaker=6025,15802")
    parser.add_argument("--bag-by", nargs="*", default=[],
                        help="Fields that define separate bags (e.g. speaker).")
    parser.add_argument("--center-within", default=None,
                        help="Field whose group mean is subtracted inside each bag (e.g. word).")
    parser.add_argument("--n-points", type=int, default=F.N_POINTS)
    parser.add_argument("--pca-variance", type=float, default=0.9,
                        help="Share of variance the retained principal components explain.")
    parser.add_argument("--max-k", type=int, default=6)
    parser.add_argument("--min-bag", type=int, default=12)
    parser.add_argument("-o", "--out", default=None)
    parser.add_argument("--name", default="split_clustering")
    args = parser.parse_args()

    pattern = re.compile(args.stem_regex)
    filters = {}
    for spec in args.keep or []:
        field, values = spec.split("=", 1)
        filters[field] = set(values.split(","))

    records, skipped = [], []
    for directory in args.audio:
        for wav in sorted(Path(directory).glob("*.wav")):
            match = pattern.match(wav.stem)
            if not match:
                continue
            fields = match.groupdict()
            if any(fields.get(f) not in v for f, v in filters.items()):
                continue
            grid = wav.with_suffix(".TextGrid")
            try:
                records.append({"stem": wav.stem, **fields,
                                **segments(wav, grid, fields["word"], args.n_points)})
            except (ValueError, OSError, KeyError) as error:
                skipped.append((wav.name, f"{type(error).__name__}: {error}"))
    if not records:
        raise SystemExit("No recording matched.")
    data = pd.DataFrame(records)
    print(f"{len(data)} recordings, {len(skipped)} skipped; "
          f"target word utterance-final in {int(data['target_is_final'].sum())} of {len(data)}")

    bags = data.groupby(args.bag_by) if args.bag_by else [("all", data)]
    rows, summary = [], []
    for bag, group in bags:
        labels = group["label"].to_numpy()
        if len(group) < args.min_bag or len(set(labels)) < 2:
            print(f"  bag {bag}: {len(group)} recordings, labels {sorted(set(labels))} -- skipped")
            continue
        ks = range(1, min(args.max_k, len(group) // 3) + 1)
        for segment in ("pre", "post"):
            for centering in ("corpus_mean", "per_utterance"):
                features = F.center(np.vstack(group[segment].to_numpy()), centering)
                features = F.fill_edge_nans(features)
                if args.center_within:
                    within = group[args.center_within].to_numpy()
                    for value in set(within):
                        rows_of = within == value
                        features[rows_of] -= features[rows_of].mean(axis=0)
                features, p = principal_scores(features, args.pca_variance)
                table = evaluate(features, labels, ks)
                meta = {"bag": str(bag), "segment": segment, "centering": centering,
                        "n": len(group), "components": p}
                rows.append(table.assign(**meta))
                choice = selected(table)
                ari2 = float(table.loc[table["k"] == 2, "ari_against_labels"].iloc[0])
                summary.append({**meta, **{f"k_{c}": k for c, k in choice.items()},
                                "ari_k2": ari2})

    out = Path(args.out) if args.out else Path(config.get("analysis_root"))
    out.mkdir(parents=True, exist_ok=True)
    pd.concat(rows, ignore_index=True).to_csv(out / f"{args.name}.csv", index=False)
    summary = pd.DataFrame(summary)
    summary.to_csv(out / f"{args.name}_summary.csv", index=False)
    if skipped:
        pd.DataFrame(skipped, columns=["file", "reason"]).to_csv(
            out / f"{args.name}_skipped.csv", index=False)

    print(summary.round(3).to_string(index=False))
    print("\nAcross bags, by segment and centering: share selecting k = 2 (gap rule, BIC), "
          "and mean ARI of the k = 2 partition against the labels")
    agg = summary.groupby(["segment", "centering"]).agg(
        bags=("bag", "size"),
        gap_k2=("k_gap_tibshirani", lambda k: float(np.mean(k == 2))),
        gap_k1=("k_gap_tibshirani", lambda k: float(np.mean(k == 1))),
        bic_k2=("k_gmm_bic", lambda k: float(np.mean(k == 2))),
        bic_k1=("k_gmm_bic", lambda k: float(np.mean(k == 1))),
        ari_k2=("ari_k2", "mean"))
    print(agg.round(3).to_string())


if __name__ == "__main__":
    main()
