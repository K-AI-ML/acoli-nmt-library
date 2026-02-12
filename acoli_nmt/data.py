"""Data loading, normalisation, deduplication, and splitting."""

from __future__ import annotations

import glob
import json
import os
import subprocess
from typing import TYPE_CHECKING

import pandas as pd
from datasets import Dataset, DatasetDict, concatenate_datasets, load_dataset

if TYPE_CHECKING:
    from acoli_nmt.config import Config

# ── Column-name candidates across all datasets ─────────────────────────────

EN_KEYS = [
    "English", "en", "english", "eng", "source",
    "text_en", "input", "src", "English_text",
]
ACH_KEYS = [
    "Acholi", "ach", "acholi", "Acoli", "acoli", "aco",
    "target", "text_ach", "Luo", "luo", "tgt", "Acholi_text",
]


def _pick(row: dict, keys: list[str]) -> str:
    for k in keys:
        v = row.get(k)
        if v is not None:
            s = str(v).strip()
            if s and s.lower() not in ("none", "nan", "<na>", ""):
                return s
    return ""


def _normalise_rows(ds: Dataset) -> Dataset:
    def _n(ex):
        return {"en": _pick(ex, EN_KEYS), "ach": _pick(ex, ACH_KEYS)}
    return ds.map(_n, remove_columns=ds.column_names)


def _is_valid(ex: dict) -> bool:
    en, ach = ex.get("en", ""), ex.get("ach", "")
    return (
        len(en) >= 2
        and len(ach) >= 2
        and en.lower().strip() != ach.lower().strip()
    )


# ── Public API ──────────────────────────────────────────────────────────────

def load_all_sources(cfg: Config) -> list[Dataset]:
    """Load and normalise all enabled data sources.  Returns list of Datasets."""
    parts: list[Dataset] = []

    if cfg.load_mt560:
        print("⏳ [1] MT560 …")
        ds = load_dataset(
            "michsethowusu/english-acholi_sentence-pairs_mt560", split="train"
        )
        print(f"   {len(ds):,} rows")
        parts.append(_normalise_rows(ds))

    if cfg.load_uglex2:
        print("⏳ [2] UgandaLex2 …")
        ds = load_dataset("allandclive/UgandaLex2", split="train")
        print(f"   {len(ds):,} rows")
        parts.append(_normalise_rows(ds))

    if cfg.load_uglex1:
        print("⏳ [3] UgandaLex v1 …")
        try:
            ds = load_dataset("allandclive/UgandaLex", split="train")
            print(f"   {len(ds):,} rows")
            parts.append(_normalise_rows(ds))
        except Exception as e:
            print(f"   ⚠️ {e}")

    if cfg.load_ugalang:
        print("⏳ [4] ugalang_0 …")
        try:
            ds = load_dataset("oumo-os/ugalang_0", split="train")
            print(f"   {len(ds):,} rows")
            parts.append(_normalise_rows(ds))
        except Exception as e:
            print(f"   ⚠️ {e}")

    # SALT — HuggingFace then GitHub
    salt_loaded = False
    if cfg.load_salt_hf:
        print("⏳ [5a] SALT (HF) …")
        try:
            ds = None
            for sp in ["text-all", "train", "text"]:
                try:
                    ds = load_dataset("Sunbird/salt", split=sp)
                    break
                except Exception:
                    ds = None
            if ds and len(ds) > 0:
                print(f"   {len(ds):,} rows")
                parts.append(_normalise_rows(ds))
                salt_loaded = True
            else:
                print("   ⚠️ no usable split")
        except Exception as e:
            print(f"   ⚠️ {e}")

    if cfg.load_salt_gh and not salt_loaded:
        sd = "/tmp/salt"
        print("⏳ [5b] SALT (GitHub) …")
        if not os.path.exists(sd):
            subprocess.run(
                ["git", "clone", "--depth", "1",
                 "https://github.com/SunbirdAI/salt.git", sd],
                check=False, capture_output=True,
            )
        rows: list[dict] = []
        for pat in ["**/*.csv", "**/*.tsv"]:
            for fp in glob.glob(os.path.join(sd, pat), recursive=True):
                try:
                    sep = "\t" if fp.endswith(".tsv") else ","
                    df0 = pd.read_csv(fp, sep=sep, on_bad_lines="skip", nrows=0)
                    cl = {c.lower(): c for c in df0.columns}
                    he = any(k in cl for k in ["english", "en", "eng"])
                    ha = any(k in cl for k in ["acholi", "ach", "acoli", "luo"])
                    if he and ha:
                        df = pd.read_csv(fp, sep=sep, on_bad_lines="skip")
                        ec = next(cl[k] for k in ["english", "en", "eng"] if k in cl)
                        ac = next(cl[k] for k in ["acholi", "ach", "acoli", "luo"] if k in cl)
                        for _, r in df.iterrows():
                            e = str(r.get(ec, "")).strip()
                            a = str(r.get(ac, "")).strip()
                            if e and a:
                                rows.append({"en": e, "ach": a})
                except Exception:
                    pass
        for pat in ["**/*.json", "**/*.jsonl"]:
            for fp in glob.glob(os.path.join(sd, pat), recursive=True):
                try:
                    with open(fp) as f:
                        data = (
                            [json.loads(line) for line in f if line.strip()]
                            if fp.endswith(".jsonl")
                            else json.load(f)
                        )
                    if isinstance(data, dict):
                        data = data.get("data", data.get("rows", [data]))
                    for it in data if isinstance(data, list) else []:
                        if not isinstance(it, dict):
                            continue
                        e, a = _pick(it, EN_KEYS), _pick(it, ACH_KEYS)
                        if e and a:
                            rows.append({"en": e, "ach": a})
                except Exception:
                    pass
        if rows:
            parts.append(Dataset.from_list(rows))
            salt_loaded = True
            print(f"   {len(rows):,} pairs")
        else:
            print("   ⚠️ none found")

    print(f"\n📦 {len(parts)} source(s) loaded")
    return parts


def clean_and_split(
    parts: list[Dataset], seed: int = 42
) -> DatasetDict:
    """Filter, deduplicate, merge, and split into train/val/test."""
    cleaned = [p.filter(_is_valid) for p in parts]
    for i, c in enumerate(cleaned):
        print(f"  Src {i + 1}: {len(c):,}")

    merged = concatenate_datasets(cleaned)
    seen: set[tuple[str, str]] = set()
    keep: list[int] = []
    for i, r in enumerate(merged):
        key = (r["en"].lower().strip(), r["ach"].lower().strip())
        if key not in seen:
            seen.add(key)
            keep.append(i)
    merged = merged.select(keep)
    print(f"\n✅ {len(merged):,} unique pairs")

    merged = merged.shuffle(seed=seed)
    n = len(merged)
    nt = max(500, int(n * 0.05))
    nv = max(500, int(n * 0.05))
    ntr = n - nv - nt

    splits = DatasetDict(
        {
            "train": merged.select(range(ntr)),
            "validation": merged.select(range(ntr, ntr + nv)),
            "test": merged.select(range(ntr + nv, n)),
        }
    )
    for k, v in splits.items():
        print(f"  {k:12s}: {len(v):>7,}")
    return splits
