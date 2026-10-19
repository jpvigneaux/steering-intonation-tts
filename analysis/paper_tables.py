#!/usr/bin/env python3
"""Write the body's intervention table from the measurement CSVs, as a LaTeX fragment.

Like ``paper_figures.py``: no number in the table is typed by hand. For each intervention and
system it reports the normalized displacement (mean over the three reported read-outs and both
directions, with the range over those eight cells), the share of declarative and of interrogative
targets whose final-word contour rises after the intervention, and the word error rate. The share
rising is what a listener would hear: a displacement is a fraction of a gap, a flip is a change of
tune.

A recording counts as rising when its final-word excursion exceeds the paper's boundary,
``RISE_BOUNDARY`` ERB (Section 2).

A second table, ``final_word.tex``, gives where on the final word the two unsteered conditions
separate in the ten-speaker sets (``methods_comparison.py`` over each pooled set).

A third, ``level0.tex``, is the cross-sentence control for what level 0 carries: swaps and
depth-model re-runs between the assertions of two different sentences by one voice
(``intervene.py --pairing cross-sentence``, ``patch_backbone.py --pairing cross-sentence``),
scored by whether the transcript names the target's or the donor's final word.

Output: ``<out>/interventions.tex``, ``<out>/final_word.tex`` and ``<out>/level0.tex``.

Example::

    python analysis/paper_tables.py --out <paper>/tables
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from interventions import config  # noqa: E402

#: Set by ``main``: the CSM instrumented set's and the S2 set's tables.
SYSTEMS: dict[str, Path] = {}
#: The rise/fall boundary on the excursion, in ERB (Section 2 and the dead-band appendix).
RISE_BOUNDARY = 0.38

#: The read-outs the paper reports: the excursion, the velocity, and the cluster read-out centred
#: on each speaker's own mean (Section 2).
REPORTED = ("A:excursion", "A:velocity", "B:cluster[per_speaker]")


def reported(table: pd.DataFrame) -> pd.DataFrame:
    """The rows of a scoring table for the read-outs the paper reports.

    A table scored before the per-speaker read-out existed has no row for it. Such tables are the
    single-voice CSM sets, for which per-speaker centring *is* the corpus mean
    (``tests/test_discriminate.py::test_per_speaker_is_the_corpus_mean_for_one_voice``), so the
    corpus-mean row is taken under the per-speaker name. A multi-voice table must carry its own.
    """
    families = set(table["family"])
    if "B:cluster[per_speaker]" not in families:
        table = table.assign(family=table["family"].replace(
            {"B:cluster[corpus_mean]": "B:cluster[per_speaker]"}))
    return table[table["family"].isin(REPORTED)]


#: (row label, {system: (source, intervention label, alpha or None)})
ROWS = [
    ("Swap levels $1$--$K{-}1$", {"CSM": ("swap", "swap-L1-31-offset", None),
                                   "S2": ("swap", "swap-L1-9-offset", None)}),
    ("Swap level $0$", {"CSM": ("swap", "swap-L0-offset", None),
                        "S2": ("swap", "swap-L0-offset", None)}),
    ("Re-run, own $\\vh_t$ and $q^0_t$", {"CSM": ("patch", "backbone-null", None),
                                      "S2": ("patch", "backbone-null", None)}),
    ("Source $\\vh_t$, own $q^0_t$", {"CSM": ("patch", "backbone-patch-a100", None),
                                  "S2": ("patch", "backbone-patch-a100", None)}),
    ("Own $\\vh_t$, source $q^0_t$", {"CSM": ("patch", "backbone-q0patch", None),
                                  "S2": ("patch", "backbone-q0patch", None)}),
    ("Source $\\vh_t$ and $q^0_t$", {"CSM": ("patch", "backbone-patch-both", None),
                                 "S2": ("patch", "backbone-patch-both", None)}),
    ("Steer, probe direction$^{\\ddagger}$", {"CSM": ("probe", None, None),
                                               "S2": ("probe", None, None)}),
    # The offline steering rows are the pre-norm edit norm(x + alpha w) since 25 September 2026
    # (analysis/prenorm_directions.py; scored into <root>/prenorm-score). Alphas are in the
    # residual's own units: 44 is 4 of CSM's gaps (10.7 to 11.4 over the held-out sentences), 678
    # is 4 of S2's (152 to 181, the two massive coordinates excluded).
    ("Steer, mean difference$^{\\S}$", {"CSM": ("steer", "backbone-steerdmpn-ap4400", 44.0),
                                         "S2": ("steer", "backbone-steerdmpn-ap67800", 678.0)}),
]


def read_many(*paths) -> pd.DataFrame:
    return pd.concat([pd.read_csv(p) for p in paths if Path(p).is_file()], ignore_index=True)


def rising(raw: pd.DataFrame, label: str) -> tuple[float, float]:
    """Share of declarative and of interrogative targets rising after ``label``."""
    rows = raw[raw["intervention"] == label]
    wide = rows.pivot_table(index=["stem", "direction"], columns="region", values="after")
    wide = wide.reset_index()
    rises = (wide["offset"] - wide["onset"]) > RISE_BOUNDARY
    return (float(rises[wide["direction"] == "A"].mean()),
            float(rises[wide["direction"] == "I"].mean()))


def unmodified_rising(raw: pd.DataFrame, label: str) -> tuple[float, float]:
    rows = raw[raw["intervention"] == label]
    wide = rows.pivot_table(index=["stem", "direction"], columns="region", values="before")
    wide = wide.reset_index()
    rises = (wide["offset"] - wide["onset"]) > RISE_BOUNDARY
    return (float(rises[wide["direction"] == "A"].mean()),
            float(rises[wide["direction"] == "I"].mean()))


def cell(system: str, spec) -> dict:
    root = SYSTEMS[system]
    source, label, alpha = spec
    swap = pd.read_csv(root / "swap_effect.csv")
    patch = read_many(root / "backbone_patch.csv")
    dose = read_many(root / "backbone_dose.csv", root / "alphafill" / "backbone_dose.csv")
    raw_swap = pd.read_csv(root / "swap_positional_raw.csv")
    raw_patch = read_many(root / "backbone_positional_raw.csv",
                          root / "alphafill" / "backbone_positional_raw.csv")
    wer = read_many(root / "wer.csv", root / "backbone" / "wer.csv",
                    root / "backbone-alphafill" / "wer.csv")
    if source in ("probe", "steer"):
        # The pre-norm offline edit, scored separately (see ROWS).
        dose = read_many(root / "prenorm-score" / "backbone_dose.csv")
        raw_patch = read_many(root / "prenorm-score" / "backbone_positional_raw.csv")
        wer = read_many(root / "backbone-prenorm" / "wer.csv")
    wer = wer.drop_duplicates(subset=["stem", "intervention"]).groupby("intervention")["wer"].mean()

    if source == "probe":
        arm = reported(dose[dose["direction_method"] == "probe"])
        by_alpha = arm.groupby("alpha")["displacement"].mean()
        best = float(by_alpha.idxmax())
        cells = arm[arm["alpha"] == best]["displacement"]
        label = f"backbone-steerpn-ap{int(round(best * 100)):03d}"
    elif source == "steer":
        arm = reported(dose[dose["direction_method"] == "diffmeans"])
        cells = arm[arm["alpha"] == alpha]["displacement"]
    else:
        table = swap if source == "swap" else patch
        cells = reported(table[table["intervention"] == label])["displacement"]
    raw = raw_swap if source == "swap" else raw_patch
    up_a, up_i = rising(raw, label)
    return {"mean": float(cells.mean()), "lo": float(cells.min()), "hi": float(cells.max()),
            "rise_a": up_a, "rise_i": up_i, "wer": float(wer.get(label, np.nan))}


def num(value: float, digits: int = 2) -> str:
    """Rounded, with a rounded zero printed unsigned: -0.001 is "0.00", not "-0.00"."""
    return f"{round(value, digits) + 0.0:.{digits}f}"


def fmt(value: float, digits: int = 2) -> str:
    return "---" if not np.isfinite(value) else f"${num(value, digits)}$"


#: The cross-sentence variants, in the order the table gives them.
CROSS_SENTENCE = [
    ("xsent-L0-offset", "swap level $0$ (codec)"),
    ("xsent-L1-31-offset", "swap levels $1$--$31$ (codec)"),
    ("backbone-q0patch", "depth model: own $\\vh_t$, source's $q^0_t$"),
    ("backbone-patch-a100", "depth model: source's $\\vh_t$, own $q^0_t$"),
    ("backbone-patch-both", "depth model: source's $\\vh_t$ and $q^0_t$"),
]


def heard(hypothesis, name: str) -> bool:
    """Whether a transcript contains the name, allowing the recognizer's spellings (Madeleine)."""
    return name[:4].lower() in str(hypothesis).lower()


def level0(ten: Path, out: Path) -> None:
    """The cross-sentence control: which sentence's final word the transcript names."""
    root = ten / "pooled"
    wer = read_many(root / "csm-xsent-analysis" / "wer.csv",
                    root / "csm-xsent-analysis" / "backbone" / "wer.csv")
    lines = [r"\begin{tabular}{@{}l r ccc@{}}", r"\toprule",
             r"CSM, final word of two sentences & $n$ & WER & target's name & source's name \\",
             r"\midrule"]
    base = wer[wer["intervention"] == "unmodified"]
    lines.append(f"unmodified & ${len(base)}$ & ${num(base['wer'].mean())}$ & "
                 f"${num(np.mean([heard(h, r.split()[-1]) for h, r in zip(base['hypothesis'], base['reference'])]))}$"
                 r" & --- \\")
    for label, name in CROSS_SENTENCE:
        manifest = next((root / "csm-xsent-decoded").glob(f"**/{label}/manifest.csv"))
        plan = pd.read_csv(manifest)[["stem", "name", "donor_stem"]]
        plan["donor_name"] = plan["donor_stem"].str.split("-").str[1]
        rows = wer[wer["intervention"] == label].merge(plan, on="stem")
        own = np.mean([heard(h, n) for h, n in zip(rows["hypothesis"], rows["name"])])
        donor = np.mean([heard(h, n) for h, n in zip(rows["hypothesis"], rows["donor_name"])])
        lines.append(f"{name} & ${len(rows)}$ & ${num(rows['wer'].mean())}$ & ${num(own)}$ & "
                     f"${num(donor)}$ \\\\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    (out / "level0.tex").write_text("\n".join(lines) + "\n")


def final_word(ten: Path, out: Path) -> None:
    """Separation of the unsteered conditions at the final word's onset and offset."""
    lines = [r"\begin{tabular}{@{}l rr rr@{}}", r"\toprule",
             r" & \multicolumn{2}{c}{CSM} & \multicolumn{2}{c}{S2} \\",
             r"\cmidrule(lr){2-3}\cmidrule(l){4-5}",
             r"region & I$-$A & $d$ & I$-$A & $d$ \\", r"\midrule"]
    tables = {system: pd.read_csv(ten / "pooled" / f"{system}-methods" /
                                  "positional_separation.csv").set_index("region")
              for system in ("csm", "s2")}
    for region in ("onset", "offset"):
        row = [region]
        for system in ("csm", "s2"):
            row += [f"${num(tables[system].loc[region, 'separation'])}$",
                    f"${num(tables[system].loc[region, 'cohens_d'])}$"]
        lines.append(" & ".join(row) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    (out / "final_word.tex").write_text("\n".join(lines) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", required=True, help="Directory to write interventions.tex into.")
    parser.add_argument("--analysis", default=None,
                        help="Analysis root (default: analysis_root in the configuration); the CSM "
                             "instrumented set's tables are read from its v2/.")
    parser.add_argument("--ten-speaker", default=None,
                        help="The ten-speaker root (default: generation.ten_speaker_root). S2's "
                             "tables are read from pooled/s2-analysis, and the final-word table "
                             "from pooled/<system>-methods.")
    args = parser.parse_args()
    analysis = Path(args.analysis) if args.analysis else config.path("analysis_root")
    ten = (Path(args.ten_speaker) if args.ten_speaker
           else config.path("generation", "ten_speaker_root"))
    # Both systems on the pooled ten-speaker sets since 25 September 2026; the CSM instrumented
    # set's tables stay in <analysis>/v2.
    SYSTEMS.update({"CSM": ten / "pooled" / "csm-analysis", "S2": ten / "pooled" / "s2-analysis"})
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    final_word(ten, out)
    level0(ten, out)

    pairs = {system: int(pd.read_csv(root / "swap_effect.csv")["n_pairs"].max())
             for system, root in SYSTEMS.items()}
    lines = [r"\begin{tabular}{@{}l rcc rcc@{}}", r"\toprule",
             rf" & \multicolumn{{3}}{{c}}{{CSM (${pairs['CSM']}$ pairs)}} "
             rf"& \multicolumn{{3}}{{c}}{{Fish S2 (${pairs['S2']}$ pairs)}} \\",
             r"\cmidrule(lr){2-4}\cmidrule(l){5-7}",
             r"Intervention & $\Delta_\Phi$ & rising, \class{A}/\class{I} & WER "
             r"& $\Delta_\Phi$ & rising, \class{A}/\class{I} & WER \\", r"\midrule"]

    base = []
    for system, root in SYSTEMS.items():
        raw = pd.read_csv(root / "swap_positional_raw.csv")
        label = "swap-L1-31-offset" if system == "CSM" else "swap-L1-9-offset"
        up_a, up_i = unmodified_rising(raw, label)
        wer = read_many(root / "wer.csv", root / "backbone" / "wer.csv")
        wer = wer[wer["intervention"] == "unmodified"]["wer"].mean()
        base += ["---", f"${up_a * 100:.0f}$/${up_i * 100:.0f}$", fmt(wer, 3)]
    lines.append("Unmodified & " + " & ".join(base) + r" \\")
    for name, specs in ROWS:
        row = [name]
        for system in SYSTEMS:
            if system not in specs:
                row += ["\\multicolumn{3}{c}{n/a}"]
                continue
            c = cell(system, specs[system])
            row += [f"${num(c['mean'])}$ {{\\scriptsize $[{num(c['lo'])}, {num(c['hi'])}]$}}",
                    f"${c['rise_a'] * 100:.0f}$/${c['rise_i'] * 100:.0f}$", fmt(c["wer"], 3)]
        lines.append(" & ".join(row) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    (out / "interventions.tex").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
