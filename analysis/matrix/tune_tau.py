"""Calibrate the soft-thresholding parameter on human data, instead of on a formula.

tau moves the null between two known regimes: at 0 it is hard thresholding, which Huang, Liu,
Yuan and Marron call "always more powerful than the sample method" but with "large potential for
creating type-I error"; at tau-tilde it conserves the total energy and is the conservative end.
Their tuning rule picks a point between by minimizing a quantity defined on the theoretical null,
which never checks what the test does on data. This measures that instead.

For each tau, given as a fraction rho of tau-tilde, the whole test is run on resamples of sets
whose structure is known or assumed, all subsampled to the size of a steering panel:

  structured   human hll + lhh (k = 2) and hll + lhh + lhl (k = 3), balanced and unbalanced
  homogeneous  one tune target, pooled; one tune target within a single sentence

The rejection rate on the homogeneous sets is the level; on the structured sets it is the power.
Neither homogeneous set is guaranteed to be one population -- a tune target pooled over ten
speakers may not be -- so the two are reported separately rather than averaged, and their
disagreement is part of the result.

Usage: tune_tau.py <rho>
"""
import sys
import numpy as np, pandas as pd

import _layout as L  # noqa: E402  -- resolves the set and puts the repository on the path
import tune_shapes as TS

RHO = float(sys.argv[1])
CORPUS = L.CORPUS
SPEAKERS = set(pd.read_csv(L.SPEAKERS_CSV)
               ["speaker"].astype(str))
SENTENCES = {"Harmony", "Madelyn", "Melanie"}
N, REPEATS, B, KS = 90, 20, 49, range(2, 5)


def wavs(tunes, sentences=SENTENCES):
    return [w for w in sorted(CORPUS.glob("*.wav"))
            if (len(w.stem.split("_")) >= 6 and w.stem.split("_")[0] in SPEAKERS
                and w.stem.split("_")[1].capitalize() in sentences
                and w.stem.split("_")[-1] in tunes and w.with_suffix(".TextGrid").is_file())]


def contours_of(paths):
    values, kept = TS.contours(paths, 30)
    return TS.centred(values, [w.stem.split("_")[0] for w in kept], "per_speaker")


def speaker_tune_means(tunes):
    """One point per speaker and tune, the unit at which the corpus's five clusters live.

    Calibrating on trials instead asks the test to find a structure that is weak there: pooled
    over speakers, single productions of eight tunes support two clusters and agree with the
    labels at 0.16, against five and 0.57 for these means. A power curve measured on trials would
    therefore be measuring the material, not the test.
    """
    paths = wavs(tunes)
    values, kept = TS.contours(paths, 30)
    speakers = np.array([w.stem.split("_")[0] for w in kept])
    labels = np.array([w.stem.split("_")[-1] for w in kept])
    scaled = TS.centred(values, list(speakers), "per_speaker")
    means = []
    for speaker in sorted(set(speakers)):
        for tune in tunes:
            rows = (speakers == speaker) & (labels == tune)
            if rows.sum():
                with np.errstate(invalid="ignore"):
                    means.append(np.nanmean(scaled[rows], axis=0))
    return np.vstack(means)


SETS = {
    "means, homogeneous: hll": (speaker_tune_means(("hll",)), 1),
    "means, homogeneous: lhh": (speaker_tune_means(("lhh",)), 1),
    "means, structured: hll+lhh": (speaker_tune_means(("hll", "lhh")), 2),
    "means, structured: hll+lhh+lhl": (speaker_tune_means(("hll", "lhh", "lhl")), 3),
    "trials, homogeneous: hll": (contours_of(wavs(("hll",))), 1),
    "trials, structured: hll+lhh": (contours_of(wavs(("hll", "lhh"))), 2),
}

rows = []
for name, (data, truth) in SETS.items():
    rng = np.random.default_rng(0)
    for repeat in range(REPEATS):
        size = min(N, len(data)) if name.startswith("trials") else len(data)
        take = rng.choice(len(data), size=size, replace=(size > len(data)))
        k, _, spoken = TS.choose_k_ch(data[take], KS, n_references=B, seed=repeat,
                                      reference="sigclust_soft", rho=RHO)
        rows.append({"rho": RHO, "set": name, "true_k": truth, "repeat": repeat,
                     "n": len(take), "k": k, "p": spoken["p"]})
    done = pd.DataFrame(rows)
    done = done[done["set"] == name]
    print(f"rho {RHO:.1f}  {name}: rejects {float((done['p'] <= 0.05).mean()):.2f}, "
          f"k chosen {done['k'].value_counts().to_dict()}", flush=True)

out = L.RESULTS / "tau_calibration"
out.mkdir(parents=True, exist_ok=True)
pd.DataFrame(rows).to_csv(out / f"rho-{RHO:.2f}.csv", index=False)
