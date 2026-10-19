"""Deciding which condition a generated recording belongs to.

An intervention needs a read-out of its own effect, and there is more than one defensible way
to build one. There are four families:

=======  ====================================  ==============  ==========  ====================
family   basis                                 supervision     probe-free  where
=======  ====================================  ==============  ==========  ====================
A        direct F0 measures over the word      none            yes         here
B        unsupervised clustering of contours   none            yes         here
C        transfer of probes trained on humans  human corpus    no          analysis/family_c_probe.py
D        linear probe on the emitted tokens    design labels   no          analysis/backbone_probe.py
=======  ====================================  ==============  ==========  ====================

The paper's **three probe-free read-outs** are A's two (excursion, velocity) and B's clustering
read-out centred on each speaker's mean (``B:cluster[per_speaker]``). :func:`default_families`
returns those and two more clustering read-outs, centred on the corpus mean and on each
utterance's own mean, which earlier drafts reported and which are kept so that their tables can be
reproduced. None of them sees a probe, a codebook or a human label, so an intervention scored by
them does not rest the argument on a probe. Family C is the human-trained HuBERT edge probe of the
paper's App. B.5, reported beside them. Family D is a decodability measurement, not a read-out of an intervened recording, and
lives with the other probes.

Every score is signed with the same orientation: **positive means interrogative-like**. That
convention is what lets an effect be averaged across families and across swap directions.

**Scoring an intervention.** The paper's displacement applies one read-out ``Phi`` to a recording
before and after the intervention. Families A and C score each recording on its own, so that is
automatic. Family B is *fitted*: its centering constant, its centroids and its orientation are
estimated from a set of recordings. :func:`score_against` fits it once on the unmodified
recordings and applies that same fit to the intervened ones, which is what the definition
requires. Calling ``score`` on the intervened set instead refits the clustering to that set, so
the read-out would change with the intervention; the pipeline did that until 15 September 2026
(``--refit-clustering`` in ``analysis/swap_effect.py`` reproduces it).
"""
from __future__ import annotations

import warnings
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Protocol

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score, roc_auc_score

from .f0 import Centering, Contour, center, fill_edge_nans, stack

#: Fraction of the nuclear word treated as its onset region, and as its offset region.
EDGE_FRACTION = 0.2
# Edge regions are fractions of the *voiced* frames, not of the grid: see ``edge_levels``.
EDGES_OVER_VOICED = True

#: Positive scores mean interrogative-like, for every family.
INTERROGATIVE = "I"
DECLARATIVE = "A"


@dataclass
class Recording:
    """One recording as a discriminator sees it."""

    stem: str
    condition: str | None            # None for material whose condition is unknown
    contour: Contour
    name: str = ""
    iteration: int = 0
    speaker: str | None = None
    meta: dict = field(default_factory=dict)

    @property
    def label(self) -> int | None:
        """1 for interrogative, 0 for declarative, None when unlabeled."""
        if self.condition is None:
            return None
        return int(self.condition == INTERROGATIVE)


class Discriminator(Protocol):
    """Scores recordings on a signed axis, positive meaning interrogative-like."""

    name: str
    probe_free: bool

    def score(self, recordings: Sequence[Recording]) -> np.ndarray:
        ...


# ---------------------------------------------------------------------------
# Family A -- direct F0 measures
# ---------------------------------------------------------------------------

def scorable(recording) -> bool:
    """Can *every* discrimination family score this recording?

    Under the voiced-fraction edge regions of ``edge_levels`` the families agree on what they
    need: some voiced material anywhere in the nuclear word. The clustering families need it
    because they are otherwise handed a row of ``NaN`` they cannot fill, and the excursion needs
    it because its edge regions are drawn from the voiced frames themselves. A word that is well
    voiced through its middle but trails off unvoiced at its end is therefore scorable, where the
    earlier grid-fraction definition excluded it.

    Families are still held to one common subset, so that a disagreement between them is a real
    disagreement and not a difference of sample. Nothing is imputed: a filled contour would enter
    the bootstrap as though it were a measurement.
    """
    if not recording.contour.is_usable:
        return False
    return bool(np.isfinite(edge_levels(recording.contour)["excursion"]))


def edge_levels(contour: Contour, edge_fraction: float = EDGE_FRACTION,
                *, over_voiced: bool = EDGES_OVER_VOICED) -> dict[str, float]:
    """Onset, offset and excursion of one nuclear-word contour, in ERB.

    ``onset`` is the mean level over the first ``edge_fraction`` of the word, ``offset`` the mean
    over the last, and ``excursion`` their difference -- the falling-or-rising direction itself.

    The decomposition is not decoration. The two conditions separate at the offset and not at
    the onset, because the punctuation manipulation moves the edge tones and leaves the pitch
    accent essentially unchanged. That converts a vague expectation into a falsifiable one: an
    intervention on the tune-bearing levels should move the offset and leave the onset alone.

    Which frames count as an "edge" is the one choice here, and it decides how often the measure
    exists at all.

    * ``over_voiced=False`` takes the edge regions as fixed fractions of the *grid*: the first
      and last ``edge_fraction`` of the 30 points. A word voiced through its middle but voiceless
      over its final fifth then has no offset level, and the excursion is undefined. On the
      generated material that excluded roughly one recording in seven.
    * ``over_voiced=True`` takes them as fractions of the *voiced* frames instead: order the
      voiced grid points by time and average the first and last ``edge_fraction`` of them. The
      measure is then defined whenever the word carries any voicing at all, which is the same
      condition the clustering families need, so the families no longer disagree about which
      recordings exist.

    The second is the default. It measures the same thing wherever the first is defined -- on a
    fully voiced word the two coincide -- and it removes an exclusion that was an artifact of the
    estimator rather than a property of the material.
    """
    trajectory = np.asarray(contour.f0_erb, dtype=float)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        if over_voiced:
            voiced = trajectory[~np.isnan(trajectory)]
            if voiced.size == 0:
                onset = offset = float("nan")
            else:
                width = max(1, int(round(voiced.size * edge_fraction)))
                onset = float(np.mean(voiced[:width]))
                offset = float(np.mean(voiced[-width:]))
        else:
            width = max(1, int(round(len(trajectory) * edge_fraction)))
            # An edge region can be entirely unvoiced -- a generation may trail off, and that is
            # data. The level is then undefined and is reported as NaN rather than imputed.
            onset = float(np.nanmean(trajectory[:width]))
            offset = float(np.nanmean(trajectory[-width:]))
    return {"onset": onset, "offset": offset, "excursion": offset - onset}


class ExcursionDiscriminator:
    """Family A: the sign of the nuclear-word excursion.

    No training, no clustering, nothing to tune. The most transparent of the four, and the one a
    reader can check by eye against a contour plot.
    """

    name = "A:excursion"
    probe_free = True

    def __init__(self, edge_fraction: float = EDGE_FRACTION):
        self.edge_fraction = edge_fraction

    def score(self, recordings: Sequence[Recording]) -> np.ndarray:
        return np.array([edge_levels(r.contour, self.edge_fraction)["excursion"]
                         for r in recordings])

    def decompose(self, recordings: Sequence[Recording]) -> pd.DataFrame:
        """Onset, offset and excursion per recording, for the positional panel."""
        rows = []
        for recording in recordings:
            row = {"stem": recording.stem, "condition": recording.condition,
                   "name": recording.name, "iteration": recording.iteration}
            row.update(edge_levels(recording.contour, self.edge_fraction))
            rows.append(row)
        return pd.DataFrame(rows)


class VelocityDiscriminator:
    """Family A, variant: mean F0 velocity over the nuclear word.

    Reported beside the excursion because it is centering-invariant -- a derivative does not care
    about absolute pitch level -- so where the two agree, the result cannot be an artifact of how
    contours were centered.
    """

    name = "A:velocity"
    probe_free = True

    def score(self, recordings: Sequence[Recording]) -> np.ndarray:
        with np.errstate(invalid="ignore"):
            return np.array([float(np.nanmean(r.contour.velocity)) for r in recordings])


# ---------------------------------------------------------------------------
# Family B -- unsupervised clustering
# ---------------------------------------------------------------------------

class ClusteringDiscriminator:
    """Family B: k-means over the resampled contours, oriented by mean excursion.

    The clustering itself is unsupervised; only its *orientation* -- which cluster is called
    interrogative -- uses a signed acoustic quantity, never the design labels. That keeps the
    adjusted Rand index against those labels an honest measure rather than a circular one.

    Three centering schemes are supported and none is a default to be forgotten:
    ``corpus_mean`` retains absolute pitch level, ``per_utterance`` retains shape alone, and
    ``per_speaker`` retains pitch level relative to each speaker's own register, which is what a
    set of several voices needs. Each recording's ``speaker`` must then be set; the speaker means
    are estimated on the reference set and applied unchanged to the scored one.

    Two ways to use it. ``score(recordings)`` fits on ``recordings`` and scores them, which is
    right for describing one set (the analysis set, one truncation depth). ``fit(reference)``
    followed by ``transform(recordings)`` applies a fit made on one set to another, which is what
    scoring an intervention needs; see :func:`score_against`. The score assumes two clusters.
    """

    probe_free = True

    def __init__(self, centering: Centering = "corpus_mean", n_clusters: int = 2,
                 random_state: int = 0):
        self.centering = centering
        self.n_clusters = n_clusters
        self.random_state = random_state
        self.name = f"B:cluster[{centering}]"
        # Set by ``fit``: the reference set's cluster labels, the grand mean subtracted under
        # corpus-mean centering, the centroids, and which centroid is the rising one.
        self.labels_: np.ndarray | None = None
        self.offset_: float | None = None
        self.speaker_offsets_: dict | None = None
        self.centroids_: np.ndarray | None = None
        self.rising_: int | None = None

    def _centered(self, recordings: Sequence[Recording]) -> np.ndarray:
        trajectories = stack([r.contour for r in recordings])
        if self.centering == "corpus_mean":
            # The reference set's grand mean, not the scored set's: a set whose register moved
            # must be seen to have moved.
            return trajectories - self.offset_
        if self.centering == "per_speaker":
            # The same reasoning, speaker by speaker.
            missing = {r.speaker for r in recordings} - set(self.speaker_offsets_)
            if missing:
                raise ValueError(f"{self.name}: no reference recordings for speakers "
                                 f"{sorted(map(str, missing))[:3]}")
            offsets = np.array([self.speaker_offsets_[r.speaker] for r in recordings])
            return trajectories - offsets[:, None]
        return center(trajectories, self.centering)

    def _features(self, recordings: Sequence[Recording]) -> np.ndarray:
        features = fill_edge_nans(self._centered(recordings))
        # fill_edge_nans leaves an all-NaN row alone, since it has nothing to hold or
        # interpolate from. Say so here rather than letting it reach the clusterer, whose
        # error names neither the recording nor the reason.
        bad = [r.stem for r, row in zip(recordings, features) if not np.isfinite(row).all()]
        if bad:
            raise ValueError(
                f"{len(bad)} recording(s) have no voiced material in the nuclear word and "
                f"cannot be clustered, e.g. {bad[:3]}. Filter with discriminate.scorable "
                f"before scoring, and report how many were dropped.")
        return features

    def fit(self, reference: Sequence[Recording]) -> "ClusteringDiscriminator":
        """Estimate the centering constant, the centroids and the orientation on ``reference``.

        The orientation is acoustic, never the design labels: the cluster whose members have
        the larger mean excursion is the interrogative-like one.
        """
        if self.centering == "corpus_mean":
            self.offset_ = float(np.nanmean(stack([r.contour for r in reference])))
        if self.centering == "per_speaker":
            # A set that names no speaker at all is one voice (the single-voice CSM sets), for
            # which per-speaker centring is the corpus mean. A set that names some but not all
            # is an error, not a voice.
            named = [r.speaker is not None for r in reference]
            if any(named) and not all(named):
                raise ValueError(f"{self.name}: some recordings have no speaker set")
            speakers = sorted({r.speaker for r in reference}, key=str)
            self.speaker_offsets_ = {
                s: float(np.nanmean(stack([r.contour for r in reference if r.speaker == s])))
                for s in speakers}
        features = self._features(reference)
        model = KMeans(n_clusters=self.n_clusters, n_init=10,
                       random_state=self.random_state).fit(features)
        self.labels_ = model.labels_
        self.centroids_ = model.cluster_centers_

        excursions = ExcursionDiscriminator().score(reference)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            means = [np.nanmean(excursions[model.labels_ == k]) for k in range(self.n_clusters)]
        self.rising_ = int(np.nanargmax(means))
        return self

    def transform(self, recordings: Sequence[Recording]) -> np.ndarray:
        """Signed distance to the declarative-like centroid, positive toward the other.

        Under the fit made by :meth:`fit`. A continuous score rather than a hard assignment, so
        an intervention that moves a recording partway is visible instead of being rounded to no
        change.
        """
        if self.centroids_ is None:
            raise RuntimeError(f"{self.name}: call fit() before transform()")
        features = self._features(recordings)
        to_rising = np.linalg.norm(features - self.centroids_[self.rising_], axis=1)
        to_falling = np.linalg.norm(features - self.centroids_[1 - self.rising_], axis=1)
        return to_falling - to_rising

    def score(self, recordings: Sequence[Recording]) -> np.ndarray:
        """Fit on ``recordings`` and score them. See :func:`score_against` for interventions."""
        return self.fit(recordings).transform(recordings)

    def cluster_ari(self, recordings: Sequence[Recording]) -> float:
        """Adjusted Rand index of the raw cluster assignment against the design labels.

        This is the quantity the clustering literature reports, and it is not the same as the
        index of the thresholded :meth:`score`: the assignment partitions by proximity in the
        full contour space, while the score collapses that to one signed axis. Both are
        reported, since they can disagree when a cluster is elongated.
        """
        self.fit(recordings)
        labels = [r.label for r in recordings]
        keep = [i for i, label in enumerate(labels) if label is not None]
        if len(keep) < 2:
            return float("nan")
        truth = [labels[i] for i in keep]
        assigned = [int(self.labels_[i]) for i in keep]
        return float(adjusted_rand_score(truth, assigned))


# ---------------------------------------------------------------------------
# Family C -- probe transfer
# ---------------------------------------------------------------------------

class ProbeTransferDiscriminator:
    """Family C: a tune probe trained on human speech, applied to generated audio.

    Connects the generated material to the linguistic categories, at the cost of no longer being
    probe-free. Held behind the same interface so the comparison table is uniform.

    The probe is supplied as a callable rather than loaded here, so this module stays free of
    any dependency on a trained-probe tree and remains testable without one.
    """

    probe_free = False

    def __init__(self, predict, name: str = "C:probe"):
        self.predict = predict
        self.name = name

    def score(self, recordings: Sequence[Recording]) -> np.ndarray:
        return np.asarray(self.predict(recordings), dtype=float)


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------

def agreement(scores: np.ndarray, labels: Sequence[int | None]) -> dict:
    """How well a signed score recovers the conditions.

    Reports the adjusted Rand index of the partition ``score > 0`` alongside **directed
    agreement** -- plain accuracy after thresholding at zero, the quantity the paper reports --
    and ``agreement``, the same accuracy with the sign ignored (the better of the two labellings).
    Keeping them together is diagnostic rather than redundant: a cell with high agreement and a
    low directed agreement is a calibration failure (an inverted sign), not an uninformative
    read-out, and reporting only the index would hide the difference.
    """
    scores = np.asarray(scores, dtype=float)
    keep = np.array([label is not None for label in labels]) & np.isfinite(scores)
    if keep.sum() < 2:
        return {"n": int(keep.sum()), "ari": np.nan, "agreement": np.nan,
                "directed_agreement": np.nan, "auc": np.nan}

    truth = np.array([label for label, k in zip(labels, keep) if k], dtype=int)
    kept = scores[keep]
    predicted = (kept > 0).astype(int)

    result = {
        "n": int(keep.sum()),
        "ari": float(adjusted_rand_score(truth, predicted)),
        "agreement": float(max((predicted == truth).mean(), (predicted != truth).mean())),
        "directed_agreement": float((predicted == truth).mean()),
    }
    result["auc"] = float(roc_auc_score(truth, kept)) if len(set(truth)) == 2 else np.nan
    return result


def compare(discriminators: Sequence[Discriminator],
            recordings: Sequence[Recording]) -> pd.DataFrame:
    """One row per family: how each recovers the conditions on the same recordings.

    This is the table that defends the dependent variable.
    """
    labels = [r.label for r in recordings]
    rows = []
    for discriminator in discriminators:
        scores = discriminator.score(recordings)
        rows.append({"family": discriminator.name,
                     "probe_free": discriminator.probe_free,
                     **agreement(scores, labels)})
    return pd.DataFrame(rows)


def score_against(family: Discriminator, reference: Sequence[Recording],
                  recordings: Sequence[Recording]) -> tuple[np.ndarray, np.ndarray]:
    """Scores of ``reference`` and of ``recordings`` under one read-out, fitted on ``reference``.

    For an intervention, ``reference`` is the unmodified recordings and ``recordings`` the
    intervened ones. A family that is fitted to data (Family B) is fitted once, on the reference,
    and that fit scores both sets, so an intervened recording is measured by exactly the read-out
    that measured its unmodified counterpart. A family that scores each recording on its own is
    simply called on both.
    """
    if hasattr(family, "fit") and hasattr(family, "transform"):
        family.fit(reference)
        return family.transform(reference), family.transform(recordings)
    return family.score(reference), family.score(recordings)


def default_families(include_probe: Discriminator | None = None,
                     edge_fraction: float = EDGE_FRACTION) -> list[Discriminator]:
    """The probe-free read-outs, plus a probe-based one when a trained probe is given.

    ``edge_fraction`` is the excursion's epsilon, exposed for the robustness sweep; the
    clustering read-outs orient themselves by the excursion's sign at the default, which the
    sweep does not need to vary.
    """
    families: list[Discriminator] = [
        ExcursionDiscriminator(edge_fraction),
        VelocityDiscriminator(),
        ClusteringDiscriminator("corpus_mean"),
        ClusteringDiscriminator("per_utterance"),
        # The one the paper reports since 25 September 2026; the other two are kept so the
        # earlier tables can be reproduced from the same run.
        ClusteringDiscriminator("per_speaker"),
    ]
    if include_probe is not None:
        families.append(include_probe)
    return families
