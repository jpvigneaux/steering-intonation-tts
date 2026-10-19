"""The in-loop steering step, on a tiny CSM model built from the model's own code.

What the in-loop experiment rests on is that its copy of ``Model.generate_frame`` *is* that
function when there is no edit -- the same arithmetic and the same random draws -- so that a run at
alpha 0 reproduces its source and any difference at alpha > 0 is the edit's. That is checked here
against the real ``generate_frame`` on a model small enough for CPU, together with the two modes'
defining property (``depth`` leaves level 0 alone) and the pipeline's bookkeeping.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT / "src"))
sys.path.insert(0, str(_ROOT / "pipeline"))
sys.path.insert(0, str(_ROOT / "analysis"))

from interventions import config, csm  # noqa: E402


def _models():
    try:
        package = config.get("generation", "csm_package", default=None)
    except config.ConfigError:  # no configuration file: a fresh clone, so no CSM source either
        package = None
    if package:
        sys.path.insert(0, str(package))
    return pytest.importorskip("models")


@pytest.fixture
def tiny():
    models = _models()
    from torchtune.models import llama3_2  # noqa: PLC0415

    def flavor():
        return llama3_2.llama3_2(vocab_size=64, num_layers=1, num_heads=2, num_kv_heads=1,
                                 embed_dim=16, max_seq_len=64, intermediate_dim=32,
                                 attn_dropout=0.0, norm_eps=1e-5, rope_base=500_000,
                                 scale_factor=32)

    models.FLAVORS["tiny"] = flavor
    torch.manual_seed(0)
    model = models.Model(models.ModelArgs(backbone_flavor="tiny", decoder_flavor="tiny",
                                          text_vocab_size=64, audio_vocab_size=60,
                                          audio_num_codebooks=4))
    torch.nn.init.normal_(model.audio_head, std=0.5)
    model.setup_caches(1)
    return model.eval()


def _prompt(n: int = 5):
    tokens = torch.zeros(1, n, 5, dtype=torch.long)
    tokens[0, :, -1] = torch.arange(1, n + 1)            # a text-only prompt
    mask = torch.zeros(1, n, 5, dtype=torch.bool)
    mask[0, :, -1] = True
    return tokens, mask, torch.arange(n).unsqueeze(0)


def _run(model, step, n_frames: int, seed: int):
    """``n_frames`` frames of the generation loop, each produced by ``step``."""
    model.reset_caches()
    torch.manual_seed(seed)
    tokens, mask, position = _prompt()
    out = []
    with torch.inference_mode():
        for _ in range(n_frames):
            sample = step(tokens, mask, position)
            out.append(sample[0].clone())
            tokens = torch.cat([sample, torch.zeros(1, 1).long()], dim=1).unsqueeze(1)
            mask = torch.cat([torch.ones_like(sample).bool(), torch.zeros(1, 1).bool()],
                             dim=1).unsqueeze(1)
            position = position[:, -1:] + 1
    return torch.stack(out)


def test_the_unedited_step_is_generate_frame(tiny):
    """Same frames, draw for draw, over a run long enough for the caches to matter."""
    reference = _run(tiny, lambda t, m, p: tiny.generate_frame(t, m, p, csm.TEMPERATURE,
                                                                 csm.TOP_K), 8, seed=3)
    copied = _run(tiny, lambda t, m, p: csm._steered_frame(tiny, t, m, p, None, "full")[0],
                  8, seed=3)
    assert torch.equal(reference, copied)


def test_a_zero_edit_changes_nothing(tiny):
    zero = torch.zeros(1, 16)
    reference = _run(tiny, lambda t, m, p: csm._steered_frame(tiny, t, m, p, None, "full")[0],
                     6, seed=5)
    edited = _run(tiny, lambda t, m, p: csm._steered_frame(tiny, t, m, p, zero, "full")[0],
                  6, seed=5)
    assert torch.equal(reference, edited)


def test_a_zero_prenorm_edit_reproduces_the_norm_exactly(tiny):
    """norm(x + 0) must be the backbone's own norm(x), bit for bit, through a whole run."""
    from interventions.depth import PrenormEdit  # noqa: PLC0415

    zero = PrenormEdit(torch.zeros(1, 16), tiny.backbone.norm.scale, tiny.backbone.norm.eps)
    reference = _run(tiny, lambda t, m, p: csm._steered_frame(tiny, t, m, p, None, "full")[0],
                     6, seed=7)
    edited = _run(tiny, lambda t, m, p: csm._steered_frame(tiny, t, m, p, zero, "full")[0],
                  6, seed=7)
    assert torch.equal(reference, edited)


def test_a_prenorm_edit_keeps_the_state_on_the_norm_surface(tiny):
    from interventions.depth import PrenormEdit  # noqa: PLC0415

    tokens, mask, position = _prompt()
    big = PrenormEdit(torch.full((1, 16), 50.0), tiny.backbone.norm.scale,
                      tiny.backbone.norm.eps)
    states = []
    for edit in (None, big):
        tiny.reset_caches()
        torch.manual_seed(2)
        with torch.inference_mode():
            residual = []
            handle = tiny.codebook0_head.register_forward_pre_hook(
                lambda _m, inputs: residual.append(inputs[0]))
            csm._steered_frame(tiny, tokens, mask, position, edit, "full")
            handle.remove()
        states.append(residual[0])
    scale = tiny.backbone.norm.scale.detach()
    for state in states:
        assert torch.allclose((state / scale).norm(), torch.tensor(16 ** 0.5), rtol=1e-3)
    assert not torch.allclose(states[0], states[1])


def test_depth_mode_leaves_level0_to_the_unedited_state(tiny):
    """On the first steered frame, with the noise in step, level 0 is the unedited draw."""
    tokens, mask, position = _prompt()
    big = torch.full((1, 16), 40.0)
    draws = {}
    for mode, edit in (("plain", None), ("depth", big), ("full", big)):
        tiny.reset_caches()
        torch.manual_seed(11)
        with torch.inference_mode():
            draws[mode] = csm._steered_frame(tiny, tokens, mask, position, edit,
                                             "full" if mode == "plain" else mode)[0][0]
    assert draws["depth"][0] == draws["plain"][0]
    assert not torch.equal(draws["full"], draws["plain"])


def test_an_unknown_mode_is_refused():
    with pytest.raises(ValueError):
        csm.generate_steered(None, "x", 0, [], prefix=np.zeros((3, 32)), onset=1, mode="half")


def test_the_comparison_names_the_first_divergent_frame():
    import steer_loop as SL  # noqa: PLC0415

    source = np.zeros((6, 2), dtype=np.int64)
    same = SL.compare(source, source.copy(), onset=2)
    assert same["identical"] and same["first_divergence"] == -1
    run = source.copy()
    run[4, 1] = 7
    moved = SL.compare(source, run, onset=2)
    assert not moved["identical"] and moved["first_divergence"] == 4
    longer = SL.compare(source, np.zeros((8, 2), dtype=np.int64), onset=2)
    assert not longer["identical"] and longer["first_divergence"] == 6


def test_loop_labels_keep_modes_strengths_and_repeats_apart():
    import steer_loop as SL  # noqa: PLC0415
    import voice_effects as VE  # noqa: PLC0415

    assert SL.label("full", 8.0) == "loopsteer-full-ap800"
    assert SL.label("null", 0.0, 1) == "loopsteer-null-ap000-r1"
    assert SL.label("full", 30.0, 0, "prenorm", "word") == "loopsteer-full-pn-word-ap3000"
    assert VE.alpha_of("loopsteer-full-ap800") == 8.0
    assert VE.alpha_of("loopsteer-full-pn-word-ap3000") == 30.0
    assert VE.alpha_of("backbone-steerdm-ap200-w025") == 2.0


def test_gap_units_scale_the_edit_by_each_directions_own_gap():
    """A grid in gaps means the same thing for two directions whose scales differ 10x."""
    import steer_loop as SL  # noqa: PLC0415

    # The label is written from the nominal alpha, so two contrasts at "2.5 gaps" share a name
    # inside their own subdirectories, while the edit applied differs with the gap.
    assert SL.label("full", 2.5, 0, "prenorm", "word") == "loopsteer-full-pn-word-ap250"
    assert SL.label("full", 2.5, 0, "prenorm", "word", "fall") == \
        "loopsteer-full-pn-word-fall-ap250"


def test_resume_keeps_a_variant_whose_manifest_covers_every_target(tmp_path):
    """A run interrupted by a full disk is continued, not repeated: the check is row count."""
    import pandas as pd  # noqa: PLC0415

    variant = tmp_path / "loopsteer-full-pn-word-ap250"
    variant.mkdir()
    pd.DataFrame([{"stem": f"s{i}"} for i in range(18)]).to_csv(variant / "manifest.csv",
                                                               index=False)
    assert len(pd.read_csv(variant / "manifest.csv")) == 18       # complete: skipped
    pd.DataFrame([{"stem": "s0"}]).to_csv(variant / "manifest.csv", index=False)
    assert len(pd.read_csv(variant / "manifest.csv")) != 18       # partial: rerun
