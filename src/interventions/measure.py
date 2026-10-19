"""Effect sizes for an intervention, with intervals, and the intelligibility control.

An intervention is judged on how far it moved the output toward the donor condition, and that
has to be reported as a magnitude with an interval rather than as a direction.

The central quantity is the paper's **normalized displacement** (its eq. for Delta_Phi). Raw
movement in ERB is not comparable across read-outs, whose scores live on different axes, so
movement is expressed as a fraction of the distance between the two conditions' baselines under
the same read-out:

    displacement = mean(score_intervened - score_unmodified)
                   / (mean baseline of the donor condition - mean baseline of the modified one)

In the paper's naming the modified recording is the *target* and the donor the *source*, so the
denominator is ``Phi_source - Phi_target``. A displacement of 1 means the intervention carried the
recording the whole way to the donor condition; 0 means it did not move it at all. The
denominator is estimated from the unmodified recordings, so it is a property of the material
rather than of the intervention.

Pairing matters: each intervened recording has an unmodified counterpart, so the numerator is
resampled by *pairs* (within-pair changes) rather than as two independent groups. The two
baseline means in the denominator are resampled alongside it, independently, so their
uncertainty is carried into the interval. :func:`cluster_displacement` is the same estimator with
the carrier sentence as the resampling unit.
"""
from __future__ import annotations

import re
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

DEFAULT_RESAMPLES = 10_000
DEFAULT_CONFIDENCE = 0.95


@dataclass
class Effect:
    """A displacement with a bootstrap interval."""

    estimate: float
    low: float
    high: float
    n: int
    confidence: float = DEFAULT_CONFIDENCE

    @property
    def excludes_zero(self) -> bool:
        """Whether the interval is consistent with no movement."""
        return not (self.low <= 0.0 <= self.high)

    def as_row(self, prefix: str = "") -> dict:
        return {f"{prefix}estimate": self.estimate, f"{prefix}low": self.low,
                f"{prefix}high": self.high, f"{prefix}n": self.n,
                f"{prefix}excludes_zero": self.excludes_zero}


def _percentile_interval(samples: np.ndarray, confidence: float) -> tuple[float, float]:
    tail = (1.0 - confidence) / 2.0
    return (float(np.nanpercentile(samples, 100 * tail)),
            float(np.nanpercentile(samples, 100 * (1.0 - tail))))


def paired_shift(before: Sequence[float], after: Sequence[float], *,
                 resamples: int = DEFAULT_RESAMPLES,
                 confidence: float = DEFAULT_CONFIDENCE,
                 seed: int = 0) -> Effect:
    """Mean within-pair change, with a bootstrap interval over pairs.

    ``before`` and ``after`` are aligned: entry *i* of each is the same recording, unmodified
    and intervened. Pairs where either value is undefined are dropped, since an unvoiced
    contour supports no comparison.
    """
    before = np.asarray(before, dtype=float)
    after = np.asarray(after, dtype=float)
    if before.shape != after.shape:
        raise ValueError(f"unpaired inputs: {before.shape} against {after.shape}")

    deltas = after - before
    deltas = deltas[np.isfinite(deltas)]
    if deltas.size < 2:
        return Effect(float("nan"), float("nan"), float("nan"), int(deltas.size), confidence)

    rng = np.random.default_rng(seed)
    draws = rng.choice(deltas, size=(resamples, deltas.size), replace=True).mean(axis=1)
    low, high = _percentile_interval(draws, confidence)
    return Effect(float(deltas.mean()), low, high, int(deltas.size), confidence)


def unpaired_difference(first: Sequence[float], second: Sequence[float], *,
                        resamples: int = DEFAULT_RESAMPLES,
                        confidence: float = DEFAULT_CONFIDENCE,
                        seed: int = 0) -> Effect:
    """Difference of means, ``second - first``, with a bootstrap interval.

    For two independent samples rather than matched pairs: the elicitation controls generate
    fresh recordings under each context condition, so nothing pairs them and
    :func:`paired_shift` would be the wrong estimator.

    The two groups are resampled independently, which is what makes the interval a statement
    about the difference of their means rather than about either group alone. ``n`` is the
    smaller group, since that is what limits the comparison.
    """
    first = np.asarray(first, dtype=float)
    second = np.asarray(second, dtype=float)
    first = first[np.isfinite(first)]
    second = second[np.isfinite(second)]
    if first.size < 2 or second.size < 2:
        return Effect(float("nan"), float("nan"), float("nan"),
                      int(min(first.size, second.size)), confidence)

    rng = np.random.default_rng(seed)
    draws = (rng.choice(second, size=(resamples, second.size), replace=True).mean(axis=1)
             - rng.choice(first, size=(resamples, first.size), replace=True).mean(axis=1))
    low, high = _percentile_interval(draws, confidence)
    return Effect(float(second.mean() - first.mean()), low, high,
                  int(min(first.size, second.size)), confidence)


def displacement(before: Sequence[float], after: Sequence[float],
                 source_baseline: Sequence[float], donor_baseline: Sequence[float], *,
                 resamples: int = DEFAULT_RESAMPLES,
                 confidence: float = DEFAULT_CONFIDENCE,
                 seed: int = 0) -> Effect:
    """Within-pair movement as a fraction of the distance between the conditions.

    The denominator -- how far apart the two conditions sit under this family, on the
    unmodified recordings -- is resampled alongside the numerator, so its own uncertainty is
    carried into the interval instead of being treated as known.
    """
    before = np.asarray(before, dtype=float)
    after = np.asarray(after, dtype=float)
    source_baseline = np.asarray(source_baseline, dtype=float)
    donor_baseline = np.asarray(donor_baseline, dtype=float)

    deltas = after - before
    deltas = deltas[np.isfinite(deltas)]
    source_baseline = source_baseline[np.isfinite(source_baseline)]
    donor_baseline = donor_baseline[np.isfinite(donor_baseline)]
    if deltas.size < 2 or source_baseline.size < 2 or donor_baseline.size < 2:
        return Effect(float("nan"), float("nan"), float("nan"), int(deltas.size), confidence)

    separation = donor_baseline.mean() - source_baseline.mean()
    if separation == 0:
        return Effect(float("nan"), float("nan"), float("nan"), int(deltas.size), confidence)

    rng = np.random.default_rng(seed)
    numerator = rng.choice(deltas, size=(resamples, deltas.size), replace=True).mean(axis=1)
    source_draws = rng.choice(source_baseline, size=(resamples, source_baseline.size),
                              replace=True).mean(axis=1)
    donor_draws = rng.choice(donor_baseline, size=(resamples, donor_baseline.size),
                             replace=True).mean(axis=1)
    with np.errstate(divide="ignore", invalid="ignore"):
        draws = numerator / (donor_draws - source_draws)

    low, high = _percentile_interval(draws, confidence)
    return Effect(float(deltas.mean() / separation), low, high, int(deltas.size), confidence)


def cohens_d(a: Sequence[float], b: Sequence[float]) -> float:
    """Standardized difference between two independent groups.

    Reported beside the displacement as a scale-free summary of how separated the conditions
    are to begin with: a large displacement over a barely separated pair of conditions means
    something different from the same displacement over a well separated one.
    """
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    a, b = a[np.isfinite(a)], b[np.isfinite(b)]
    if a.size < 2 or b.size < 2:
        return float("nan")
    pooled = np.sqrt(((a.size - 1) * a.var(ddof=1) + (b.size - 1) * b.var(ddof=1))
                     / (a.size + b.size - 2))
    return float((b.mean() - a.mean()) / pooled) if pooled else float("nan")


def flip_rates(before: Sequence[float], after: Sequence[float],
               conditions: Sequence[str]) -> dict[str, float]:
    """Share of targets whose tune an intervention turned into the other condition's.

    For a target with excursion ``x`` (ERB, positive rising), write ``R(x) = [x > 0]``. The
    condition's own tune is rising for ``I`` and falling for ``A``. A target is *eligible* when its
    unmodified recording shows its own condition's tune -- a declarative that already rises, or a
    question that does not, has nothing to flip from and is counted apart -- and it *flips* when
    ``R(after) != R(before)``. Per condition ``c``:

        flip_c = #(eligible and flipped, condition c) / #(eligible, condition c)

    and ``flip`` is the unweighted mean of ``flip_A`` and ``flip_I``, so each direction counts
    once whatever its sample size. This replaces reporting the share *rising* after an
    intervention, which conflates the two directions: for a declarative target rising is
    success, for an interrogative it is failure.
    """
    before = np.asarray(before, dtype=float)
    after = np.asarray(after, dtype=float)
    conditions = np.asarray(conditions)
    defined = np.isfinite(before) & np.isfinite(after)
    rising_before, rising_after = before > 0, after > 0
    out: dict[str, float] = {}
    for condition, own_tune_rises in (("A", False), ("I", True)):
        members = defined & (conditions == condition)
        eligible = members & (rising_before == own_tune_rises)
        flipped = eligible & (rising_after != rising_before)
        out[f"n_{condition}"] = int(members.sum())
        out[f"eligible_{condition}"] = int(eligible.sum())
        out[f"flip_{condition}"] = (float(flipped.sum() / eligible.sum()) if eligible.any()
                                    else float("nan"))
    out["flip"] = float(np.nanmean([out["flip_A"], out["flip_I"]]))
    return out


#: Everything that is not part of a word: punctuation, and apostrophes that do not sit between
#: two word characters (``won't`` keeps its apostrophe, a quotation mark does not).
_NOT_A_WORD = re.compile(r"[^\w\s']|(?<!\w)'|'(?!\w)")


def words(text: str) -> list[str]:
    """Lower-cased words with punctuation removed, the unit a word error is counted in.

    The recognizer punctuates and the prompts it is scored against do not, and it punctuates by
    what it hears: a falling ending gets a full stop and a rising one a question mark. Scoring the
    raw strings would therefore count a punctuation mark as a substituted word on almost every
    recording, and differently by condition.
    """
    return _NOT_A_WORD.sub(" ", text.lower()).split()


def word_error_rate(reference: str, hypothesis: str) -> float:
    """Levenshtein distance over words, divided by the reference length.

    The content control: segmental material is supposed to follow level 0, which a swap of the
    tune-bearing levels leaves untouched, so an intervention that preserves the attribute while
    destroying the words has not shown what it claims. Both strings are reduced by :func:`words`
    first, so only words are compared.
    """
    ref = words(reference)
    hyp = words(hypothesis)
    if not ref:
        return float("nan")

    previous = list(range(len(hyp) + 1))
    for i, ref_word in enumerate(ref, start=1):
        current = [i]
        for j, hyp_word in enumerate(hyp, start=1):
            current.append(min(previous[j] + 1,                       # deletion
                               current[j - 1] + 1,                    # insertion
                               previous[j - 1] + (ref_word != hyp_word)))
        previous = current
    return previous[-1] / len(ref)


def displacement_distribution(before: Sequence[float], after: Sequence[float],
                              source_baseline: Sequence[float],
                              donor_baseline: Sequence[float]) -> dict:
    """Per-recording displacements, summarized.

    :func:`displacement` returns a ratio of means, which is one number and says nothing about
    how the recordings are spread around it. A small mean is consistent with two very different
    situations --- every recording moved a little, or most moved not at all while a few moved a
    great deal --- and for a null result the difference matters, because the second would mean
    the intervention does something to some material and the first that it does nothing to any.

    Each recording's own displacement is its within-pair change over the *same* denominator the
    aggregate uses, so the mean of these is exactly :func:`displacement`'s estimate and the
    spread is comparable across conditions. The denominator is a property of the material and is
    not resampled here; the interval on the mean is what :func:`displacement` is for.
    """
    before = np.asarray(before, dtype=float)
    after = np.asarray(after, dtype=float)
    source_baseline = np.asarray(source_baseline, dtype=float)
    donor_baseline = np.asarray(donor_baseline, dtype=float)

    deltas = after - before
    deltas = deltas[np.isfinite(deltas)]
    source_baseline = source_baseline[np.isfinite(source_baseline)]
    donor_baseline = donor_baseline[np.isfinite(donor_baseline)]
    empty = {"per_pair_mean": float("nan"), "per_pair_median": float("nan"),
             "per_pair_sd": float("nan"), "per_pair_q25": float("nan"),
             "per_pair_q75": float("nan"), "per_pair_min": float("nan"),
             "per_pair_max": float("nan"), "per_pair_share_above_half": float("nan")}
    if deltas.size < 2 or source_baseline.size < 2 or donor_baseline.size < 2:
        return empty

    separation = donor_baseline.mean() - source_baseline.mean()
    if separation == 0:
        return empty

    per_pair = deltas / separation
    return {
        "per_pair_mean": float(per_pair.mean()),
        "per_pair_median": float(np.median(per_pair)),
        "per_pair_sd": float(per_pair.std(ddof=1)),
        "per_pair_q25": float(np.percentile(per_pair, 25)),
        "per_pair_q75": float(np.percentile(per_pair, 75)),
        "per_pair_min": float(per_pair.min()),
        "per_pair_max": float(per_pair.max()),
        # How much of the sample moved at least halfway. A null whose recordings are all near
        # zero and a null averaging over large movements of both signs are different findings.
        "per_pair_share_above_half": float(np.mean(per_pair >= 0.5)),
    }


def cluster_displacement(before: Sequence[float], after: Sequence[float],
                         pair_groups: Sequence[str],
                         source_baseline: Sequence[float], source_groups: Sequence[str],
                         donor_baseline: Sequence[float], donor_groups: Sequence[str], *,
                         resamples: int = DEFAULT_RESAMPLES,
                         confidence: float = DEFAULT_CONFIDENCE,
                         seed: int = 0) -> Effect:
    """:func:`displacement` with the *carrier sentence* as the resampling unit.

    The paired bootstrap treats the intervened recordings as independent, which they are not:
    they come from six carrier sentences, thirteen iterations each, and two recordings of the
    same sentence share its words, its nuclear vowel and its length. Resampling sentences with
    replacement -- and taking every pair and every baseline recording of a drawn sentence --
    propagates that dependence into the interval instead of assuming it away.

    Six clusters is few, so this is reported as a sensitivity check beside the paired interval
    rather than in place of it: a cluster bootstrap on six units is anticonservative in coverage,
    and an interval it widens materially is the signal worth acting on.

    The point estimate is :func:`displacement`'s; only the interval differs.

    Drawing ``K`` clusters with replacement from ``K`` is a multinomial draw over their
    multiplicities, and every quantity the statistic needs is a ratio of sums over the drawn
    clusters. So a resample's mean is ``(m . sums) / (m . counts)`` for the multiplicity vector
    ``m``, and the whole bootstrap is three matrix products rather than a loop over resamples.
    This is the same estimator, not an approximation of it.
    """
    def _finite(values, groups):
        values = np.asarray(values, dtype=float)
        groups = np.asarray(groups)
        keep = np.isfinite(values)
        return values[keep], groups[keep]

    deltas, pair_groups = _finite(np.asarray(after, dtype=float) - np.asarray(before, dtype=float),
                                  pair_groups)
    source_baseline, source_groups = _finite(source_baseline, source_groups)
    donor_baseline, donor_groups = _finite(donor_baseline, donor_groups)

    clusters = np.unique(np.concatenate([pair_groups, source_groups, donor_groups]))
    if deltas.size < 2 or clusters.size < 2:
        return Effect(float("nan"), float("nan"), float("nan"), int(deltas.size), confidence)

    separation = donor_baseline.mean() - source_baseline.mean()
    if separation == 0:
        return Effect(float("nan"), float("nan"), float("nan"), int(deltas.size), confidence)

    def _per_cluster(values, groups):
        masks = groups[None, :] == clusters[:, None]
        return masks @ values, masks.sum(axis=1).astype(float)

    delta_sums, delta_counts = _per_cluster(deltas, pair_groups)
    source_sums, source_counts = _per_cluster(source_baseline, source_groups)
    donor_sums, donor_counts = _per_cluster(donor_baseline, donor_groups)

    rng = np.random.default_rng(seed)
    multiplicity = rng.multinomial(clusters.size, np.full(clusters.size, 1.0 / clusters.size),
                                   size=resamples).astype(float)

    with np.errstate(divide="ignore", invalid="ignore"):
        numerator = (multiplicity @ delta_sums) / (multiplicity @ delta_counts)
        source_draw = (multiplicity @ source_sums) / (multiplicity @ source_counts)
        donor_draw = (multiplicity @ donor_sums) / (multiplicity @ donor_counts)
        draws = numerator / (donor_draw - source_draw)

    low, high = _percentile_interval(draws, confidence)
    return Effect(float(deltas.mean() / separation), low, high, int(deltas.size), confidence)
