# 🌍 Acoli NMT — Acoli ↔ English Neural Machine Translation

A complete library for training, evaluating, and serving Acoli (Acholi) ↔ English translation models. Built on NLLB-200 with LoRA fine-tuning, optimized for low-resource settings.

## Install

```bash
# From source
git clone https://github.com/your-username/acoli-nmt.git
cd acoli-nmt
pip install -e ".[all]"

# Or just the core (no Gradio/COMET)
pip install -e .
```

**Requirements:** Python ≥3.9, PyTorch ≥2.1, CUDA recommended (runs on CPU but slow).

## Quick Start — Python API

```python
from acoli_nmt import Config, Trainer, Translator

# ── Train from scratch ──────────────────────────────────────────────
cfg = Config(epochs=3, lr=1e-4, batch_size=16)
trainer = Trainer(cfg)
trainer.load_data()          # downloads & merges 5 parallel corpora
trainer.train()              # fine-tunes NLLB-200 with LoRA
metrics = trainer.evaluate() # bidirectional BLEU/chrF/COMET
trainer.save("./my-model")

# ── Translate with a trained model ──────────────────────────────────
t = Translator.from_pretrained("./my-model")
t.en_to_ach("How are you today?")
t.ach_to_en("Itye nining?")
t.translate_batch(["Hello", "Thank you"], direction="en2ach")
```

## CLI Commands

```bash
# Train (downloads data automatically)
acoli-train --epochs 5 --lr 1e-4 --save-dir ./my-model

# Evaluate a trained model
acoli-eval ./my-model

# Translate text
acoli-translate ./my-model "How are you today?"
acoli-translate ./my-model -d ach2en "Itye nining?"

# Pipe mode (one sentence per line)
cat sentences.txt | acoli-translate ./my-model

# Launch Gradio web UI
acoli-serve ./my-model --share
acoli-serve ./my-model --port 8080
```

## Gradio UI

After training, launch an interactive translation dashboard:

```bash
acoli-serve ./my-model --share
```

This gives you:
- **Translate tab** — type text, pick direction, get translation + optional BLEU/chrF scoring
- **Batch tab** — paste multiple sentences, translate all at once
- **Gallery tab** — run built-in examples through the model

`--share` creates a public URL (works on Colab too).

## Configuration

All settings are in the `Config` dataclass. You can pass them as constructor args:

```python
cfg = Config(
    model_path="nllb",                            # "nllb" or "llm"
    nllb_model="facebook/nllb-200-distilled-600M", # or 1.3B, 3.3B
    epochs=5,
    lr=1e-4,
    batch_size=16,
    grad_accum=4,         # effective batch = 64
    max_len=256,
    use_lora=True,
    lora_r=16,
    nllb_bidirectional=True,  # train both en→ach and ach→en
)
```

Or load from YAML:

```python
import yaml
cfg = Config(**yaml.safe_load(open("configs/default.yaml")))
```

## Data Sources

| Dataset | Size | License |
|---------|------|---------|
| MT560 English–Acholi | ~73K pairs | CC-BY-4.0 |
| UgandaLex v1 | ~6.2K | CC-BY-4.0 |
| UgandaLex v2 | ~6.2K | CC-BY-4.0 |
| SALT (Sunbird) | ~25K | CC-BY-SA-4.0 |
| ugalang_0 | ~2.6K | See dataset card |

All downloaded automatically on first run.

## Architecture

- **Base model:** NLLB-200-distilled-600M (seq2seq, 202 languages)
- **New language:** `ach_Latn` added via `NllbTokenizer(language_codes=...)` with embedding initialized from related `luo_Latn`
- **Fine-tuning:** LoRA (r=16, α=32) on q_proj + v_proj
- **Training:** Bidirectional (en→ach + ach→en), cosine LR, early stopping on chrF
- **Evaluation:** BLEU + chrF + COMET (optional), both directions
- **Optional:** Back-translation with round-trip filtering (chrF ≥0.65), dynamic curriculum sampling

## Hardware

| Setup | GPU | Works? |
|-------|-----|--------|
| NLLB-600M + LoRA | T4 (16GB) | ✅ |
| NLLB-600M + LoRA | RTX 3090/4090 | ✅ |
| NLLB-1.3B + LoRA | A10/A100 | ✅ |
| Tower-7B + QLoRA | T4 (16GB) | ✅ (tight) |
| CPU only | — | ⚠️ Very slow |

If you hit OOM: reduce `max_len` → 192 or `grad_accum` → 3.

## License

MIT. Training data has its own licenses (see table above).
