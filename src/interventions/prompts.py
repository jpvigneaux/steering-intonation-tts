"""Voice prompts drawn from the ten-speaker prompt plan.

``pipeline/plan_prompts.py`` gives each voice, for each carrier sentence, a few prompts of eight
human recordings -- one per nuclear tune, in random order, none of them of the sentence to be
said -- so that the prompt carries the speaker's voice and no single tune. The plan ships in
``data/ten-speaker-plan/prompt_plan.csv``.

Every generator conditions on the same prompts, CSM and S2 alike, so a cell's prompt is read
from the plan here rather than rebuilt by each script.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from .alignment import transcript

#: A prompt: the recordings given to the model, in order, each with its transcript.
Prompt = tuple[tuple[Path, str], ...]


def planned_prompt(plan: pd.DataFrame, corpus: Path, voice: str, sentence: str,
                   prompt: int) -> tuple[Prompt, str]:
    """One cell's prompt and the label its archive records as the reference.

    ``sentence`` is the carrier sentence's nuclear word (``Harmony``); ``prompt`` counts from 1.
    Each transcript is the alignment's word tier of that recording, which carries no punctuation.
    The label names the prompt and its tunes in order, e.g. ``prompt-2:lhh,hll,...``.
    """
    rows = plan[(plan["voice"] == voice) & (plan["sentence"] == sentence)
                & (plan["prompt"] == prompt)].sort_values("position")
    if rows.empty:
        raise ValueError(f"no prompt rows for {voice} {sentence} prompt {prompt}")
    corpus = Path(corpus)
    recordings = tuple((corpus / name, transcript((corpus / name).with_suffix(".TextGrid")))
                       for name in rows["file"])
    return recordings, f"prompt-{prompt}:" + ",".join(rows["tune"])
