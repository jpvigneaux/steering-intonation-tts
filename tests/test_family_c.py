

def test_the_embedding_path_separates_voices_and_contrasts(tmp_path):
    """One variant name exists under every voice and contrast; a shared path loses two of three."""
    import pandas as pd

    cache = tmp_path / "cache"
    rows = [{"voice": "d-1", "contrast": c, "variant": "loopsteer-full-pn-word-ap250",
             "stem": "sent-Harmony-A-iter-1"} for c in ("tune", "edge", "accent")]
    paths = {str(cache.joinpath(*[str(r[k]) for k in ("voice", "contrast")],
                                r["variant"], f"{r['stem']}.pt")) for r in rows}
    assert len(paths) == 3
