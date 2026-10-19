"""Per-frame token archives: every quantizer level of every sampled frame.

An archive records what a generation run actually sampled. This has to be stored at generation
time rather than replayed from a seed: sampling draws from a renormalized top-k distribution at
a non-zero temperature, so a run is not reproducible from the prompt alone, and bit-exact replay
would additionally require the same device, dtype and kernel selection. The seed is kept as a
provenance record, not as a substitute.

Layout on disk, one ``.npz`` per recording plus one index CSV for the set:

===============  ==============================  ===================================
key              shape                           holds
===============  ==============================  ===================================
``tokens``       ``(T, n_levels)`` int           every quantizer level of every frame
``c0``           ``(T,)`` int                    the sampled level-0 token
``c0_logits``    ``(T, vocab)`` float32          the level-0 head's output, optional
``last_h``       ``(T, d_model)`` float32        backbone final hidden state, optional
scalars          --                              stem, name, sentence, condition,
                                                 iteration, frame rate; optionally seed,
                                                 sample rate, temperature, top_k, top_p
===============  ==============================  ===================================

The sampling settings (``temperature``, ``top_k``, ``top_p``) are recorded by every archive
written since they were added to this format. The CSM instrumented set (156 recordings) and the
S2 set (36) predate that and carry the seed only; they were drawn at the module constants
``csm.TEMPERATURE``/``csm.TOP_K`` and ``fish_s2.TEMPERATURE``/``TOP_K``/``TOP_P``. The 36-recording
CSM pilot set stores them under the legacy key ``topk``, which :func:`load` also reads.

``n_levels`` is always read from the array and never assumed: the Mimi-based system has 32
levels, the second system has 10.
"""
from __future__ import annotations

from collections.abc import Iterator, Sequence
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

INDEX_NAME = "generation_index.csv"
LATENTS_DIR = "latents"

#: Columns the index carries, in order.
INDEX_COLUMNS = ("stem", "name", "condition", "iteration", "n_frames",
                 "duration_s", "seed", "sentence")

#: Written when any archive in the set carries them; absent from the sets that predate voices,
#: which is why they are not required on reading.
VOICE_COLUMNS = ("voice", "reference")

#: The punctuation conditions. ``A`` is the declarative and ``I`` the interrogative, the two the
#: causal results use. ``N`` is the neutral prompt, whose text ends in no punctuation at all: it
#: carries no orthographic cue to the tune, so a steering edit there works against nothing.
CONDITIONS = ("A", "I", "N")


@dataclass
class Archive:
    """One generated recording, with every quantizer level of every frame."""

    stem: str
    name: str                     # the sentence's short name, e.g. "Marilyn"
    condition: str                # "A" or "I"
    iteration: int
    sentence: str                 # the full prompt text
    tokens: np.ndarray            # (T, n_levels) int
    seed: int | None = None
    frame_rate_hz: float = 12.5
    sample_rate: int | None = None
    temperature: float | None = None
    top_k: int | None = None
    top_p: float | None = None    # S2 only; CSM samples without nucleus truncation
    voice: str | None = None      # the set's voice tag; None for single-voice sets
    reference: str | None = None  # stem of the recording that conditioned the voice, never a path
    c0_logits: np.ndarray | None = field(default=None, repr=False)
    last_h: np.ndarray | None = field(default=None, repr=False)

    def __post_init__(self) -> None:
        if self.tokens.ndim != 2:
            raise ValueError(f"{self.stem}: tokens must be (T, n_levels), got {self.tokens.shape}")
        if self.condition not in CONDITIONS:
            raise ValueError(f"{self.stem}: condition {self.condition!r} not in {CONDITIONS}")

    @property
    def n_frames(self) -> int:
        return int(self.tokens.shape[0])

    @property
    def n_levels(self) -> int:
        """Number of quantizer levels. Read from the data, never assumed."""
        return int(self.tokens.shape[1])

    @property
    def duration_s(self) -> float:
        return self.n_frames / self.frame_rate_hz

    @property
    def c0(self) -> np.ndarray:
        """The level-0 token sequence -- the level the backbone samples directly."""
        return self.tokens[:, 0]

    def with_tokens(self, tokens: np.ndarray, stem: str | None = None) -> "Archive":
        """A copy carrying different tokens, for writing an intervention's output.

        The per-frame model internals are dropped: they describe the original sampling run and
        would be misleading beside tokens that run never produced. The seed and sampling
        settings are kept as provenance of the recording the new tokens were derived from.
        """
        return Archive(
            stem=stem or self.stem, name=self.name, condition=self.condition,
            iteration=self.iteration, sentence=self.sentence, tokens=np.asarray(tokens),
            seed=self.seed, frame_rate_hz=self.frame_rate_hz, sample_rate=self.sample_rate,
            temperature=self.temperature, top_k=self.top_k, top_p=self.top_p,
            voice=self.voice, reference=self.reference,
        )


def _scalar(value: np.ndarray):
    """Unwrap a 0-d array as written by ``np.savez``."""
    item = value.item() if getattr(value, "ndim", None) == 0 else value
    return item.decode() if isinstance(item, bytes) else item


def load(path: Path, *, with_internals: bool = False) -> Archive:
    """Read one archive.

    ``with_internals`` also loads ``c0_logits`` and ``last_h``, which dominate the file size and
    are needed only by analyses of the backbone itself.
    """
    path = Path(path)
    with np.load(path, allow_pickle=False) as data:

        def optional(key: str, cast):
            return cast(_scalar(data[key])) if key in data else None

        tokens = np.asarray(data["tokens"])
        stem = str(_scalar(data["stem"])) if "stem" in data else path.stem
        archive = Archive(
            stem=stem,
            name=str(_scalar(data["name"])) if "name" in data else _name_from_stem(stem),
            condition=str(_scalar(data["condition"])),
            iteration=int(_scalar(data["iteration"])),
            sentence=str(_scalar(data["sentence"])),
            tokens=tokens,
            seed=optional("seed", int),
            frame_rate_hz=optional("frame_rate_hz", float) or 12.5,
            sample_rate=optional("sample_rate", int),
            temperature=optional("temperature", float),
            # "topk" is the key the pilot set was written with.
            top_k=optional("top_k", int) if "top_k" in data else optional("topk", int),
            top_p=optional("top_p", float),
            voice=optional("voice", str),
            reference=optional("reference", str),
            c0_logits=np.asarray(data["c0_logits"]) if with_internals and "c0_logits" in data else None,
            last_h=np.asarray(data["last_h"]) if with_internals and "last_h" in data else None,
        )
    return archive


def _name_from_stem(stem: str) -> str:
    """Recover the sentence name from a ``sent-<Name>-<A|I>-iter-<i>`` stem."""
    parts = stem.split("-")
    return parts[1] if len(parts) > 1 else stem


def save(archive: Archive, path: Path) -> None:
    """Write one archive, creating the parent directory if needed."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload: dict[str, np.ndarray] = {
        "tokens": np.asarray(archive.tokens),
        "c0": np.asarray(archive.c0),
        "stem": np.array(archive.stem),
        "name": np.array(archive.name),
        "condition": np.array(archive.condition),
        "iteration": np.array(archive.iteration),
        "sentence": np.array(archive.sentence),
        "frame_rate_hz": np.array(archive.frame_rate_hz),
    }
    for key in ("seed", "sample_rate", "temperature", "top_k", "top_p", "voice", "reference"):
        value = getattr(archive, key)
        if value is not None:
            payload[key] = np.array(value)
    if archive.c0_logits is not None:
        payload["c0_logits"] = archive.c0_logits
    if archive.last_h is not None:
        payload["last_h"] = archive.last_h
    np.savez_compressed(path, **payload)


def load_index(root: Path) -> pd.DataFrame:
    """Read a generation set's index, with an ``archive_path`` column resolved."""
    root = Path(root)
    index_csv = root / INDEX_NAME
    if not index_csv.is_file():
        raise FileNotFoundError(
            f"{index_csv} not found. Point --archives at a generation set root, which holds "
            f"{INDEX_NAME} beside a {LATENTS_DIR}/ directory.")
    df = pd.read_csv(index_csv)
    missing = set(INDEX_COLUMNS) - set(df.columns)
    if missing:
        raise ValueError(f"{index_csv} is missing column(s): {sorted(missing)}")
    df["archive_path"] = df["stem"].map(lambda s: root / LATENTS_DIR / f"{s}.npz")
    return df


def load_set(root: Path, *, with_internals: bool = False) -> list[Archive]:
    """Every archive of a generation set, in index order."""
    index = load_index(root)
    return [load(p, with_internals=with_internals) for p in index["archive_path"]]


def write_index(archives: Sequence[Archive], root: Path) -> Path:
    """Write the index CSV for a set of archives."""
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    rows = [{
        "stem": a.stem, "name": a.name, "condition": a.condition, "iteration": a.iteration,
        "n_frames": a.n_frames, "duration_s": round(a.duration_s, 4),
        "seed": a.seed if a.seed is not None else -1, "sentence": a.sentence,
        "voice": a.voice, "reference": a.reference,
    } for a in archives]
    columns = list(INDEX_COLUMNS)
    if any(a.voice is not None or a.reference is not None for a in archives):
        columns += list(VOICE_COLUMNS)
    out = root / INDEX_NAME
    pd.DataFrame(rows, columns=columns).to_csv(out, index=False)
    return out


def pairs(archives: Sequence[Archive]) -> Iterator[tuple[Archive, Archive]]:
    """Yield ``(declarative, interrogative)`` archives of the same voice, sentence and iteration.

    The matched pair is the unit of the swap experiment: the two recordings share a sentence, a
    voice and a sampling iteration, and differ only in the punctuation that ends the prompt. That
    is what makes a level exchange between them interpretable, and what distinguishes it from
    exchanging levels between unrelated utterances.

    The voice is part of the key. Without it, a set holding two voices would group four
    recordings under one (sentence, iteration), keep whichever declarative and interrogative came
    last, and pair one voice's statement with another's question -- silently, since the swap and
    its scoring run just the same on the mismatched pair. Sets that predate voices carry
    ``voice = None`` throughout and pair exactly as before.
    """
    by_key: dict[tuple[str, str, int], dict[str, Archive]] = {}
    for archive in archives:
        key = (archive.voice or "", archive.name, archive.iteration)
        group = by_key.setdefault(key, {})
        if archive.condition in group:
            raise ValueError(f"two {archive.condition} recordings share {key}: "
                             f"{group[archive.condition].stem} and {archive.stem}")
        group[archive.condition] = archive
    for key in sorted(by_key):
        group = by_key[key]
        if "A" in group and "I" in group:
            yield group["A"], group["I"]


def summarize(archives: Sequence[Archive]) -> pd.DataFrame:
    """One row per recording: the shape facts an analysis needs before loading arrays."""
    return pd.DataFrame([{
        "stem": a.stem, "name": a.name, "condition": a.condition, "iteration": a.iteration,
        "n_frames": a.n_frames, "n_levels": a.n_levels, "duration_s": a.duration_s,
    } for a in archives])
