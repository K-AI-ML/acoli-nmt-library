"""Evaluation metrics: BLEU, chrF, and optional COMET."""

from __future__ import annotations

import evaluate


_bleu = evaluate.load("sacrebleu")
_chrf = evaluate.load("chrf")

# COMET loaded lazily (large download)
_comet_model = None
_comet_available: bool | None = None


def _load_comet():
    global _comet_model, _comet_available
    if _comet_available is not None:
        return _comet_available
    try:
        from comet import download_model, load_from_checkpoint

        path = download_model("Unbabel/wmt22-comet-da")
        _comet_model = load_from_checkpoint(path)
        _comet_available = True
        print("✓ COMET loaded")
    except Exception:
        _comet_available = False
        print("ℹ️ COMET unavailable")
    return _comet_available


def score_all(
    predictions: list[str],
    references: list[str],
    sources: list[str] | None = None,
    use_comet: bool = True,
) -> dict[str, float]:
    """Compute BLEU + chrF + optional COMET."""
    refs_nested = [[r] for r in references]
    out = {
        "bleu": round(
            _bleu.compute(predictions=predictions, references=refs_nested)["score"], 2
        ),
        "chrf": round(
            _chrf.compute(predictions=predictions, references=refs_nested)["score"], 2
        ),
    }
    if use_comet and sources and _load_comet() and _comet_model:
        import torch

        ci = [
            {"src": s, "mt": p, "ref": r}
            for s, p, r in zip(sources, predictions, references)
        ]
        gpu = 1 if torch.cuda.is_available() else 0
        out["comet"] = round(
            _comet_model.predict(ci, batch_size=32, gpus=gpu).system_score, 4
        )
    return out


def sentence_score(hypothesis: str, reference: str) -> dict[str, float]:
    """Quick per-sentence BLEU + chrF."""
    refs = [[reference]]
    return {
        "bleu": round(
            _bleu.compute(predictions=[hypothesis], references=refs)["score"], 2
        ),
        "chrf": round(
            _chrf.compute(predictions=[hypothesis], references=refs)["score"], 2
        ),
    }
