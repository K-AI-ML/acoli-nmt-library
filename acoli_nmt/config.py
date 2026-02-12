"""Configuration dataclass for the Acoli NMT pipeline."""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class Config:
    """All pipeline settings in one place.  Mirrors the Colab Cell-2 toggles."""

    # ── Model ───────────────────────────────────────────────────────────
    model_path: str = "nllb"  # "nllb" or "llm"
    nllb_model: str = "facebook/nllb-200-distilled-600M"
    llm_model: str = "Unbabel/TowerInstruct-7B-v0.2"

    # ── Languages ───────────────────────────────────────────────────────
    src_lang: str = "eng_Latn"
    tgt_lang: str = "ach_Latn"
    related_lang: str = "luo_Latn"  # for embedding init

    # ── Data toggles ────────────────────────────────────────────────────
    load_mt560: bool = True
    load_uglex2: bool = True
    load_uglex1: bool = True
    load_salt_hf: bool = True
    load_salt_gh: bool = True
    load_ugalang: bool = True

    # ── Augmentation ────────────────────────────────────────────────────
    do_backtranslation: bool = False
    bt_num_samples: int = 5000
    bt_roundtrip_filter: bool = True
    bt_roundtrip_threshold: float = 0.65

    # ── Curriculum ──────────────────────────────────────────────────────
    do_curriculum: bool = False
    curriculum_oversample: int = 2
    curriculum_top_frac: float = 0.25

    # ── Training ────────────────────────────────────────────────────────
    max_len: int = 256
    use_lora: bool = True
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    batch_size: int = 16
    grad_accum: int = 4
    epochs: int = 5
    lr: float = 1e-4
    warmup: float = 0.10
    seed: int = 42
    nllb_bidirectional: bool = True

    # ── Output ──────────────────────────────────────────────────────────
    output_dir: str = "./acoli-en-nllb-ft"
    save_dir: str = "./acoli-en-nllb-final"

    # ── Device ──────────────────────────────────────────────────────────
    device: Optional[str] = None  # auto-detected if None

    @property
    def effective_batch(self) -> int:
        return self.batch_size * self.grad_accum

    def summary(self) -> str:
        return (
            f"Model={self.model_path.upper()}  LoRA={self.use_lora}  "
            f"Batch={self.effective_batch}  LR={self.lr}  "
            f"MaxLen={self.max_len}  Bidir={self.nllb_bidirectional}"
        )
