"""End-to-end Trainer: data → model → train → evaluate → save."""

from __future__ import annotations

from typing import Optional

import numpy as np
import torch
from datasets import DatasetDict, concatenate_datasets

from acoli_nmt.config import Config
from acoli_nmt.data import clean_and_split, load_all_sources
from acoli_nmt.metrics import score_all
from acoli_nmt.models import apply_lora, get_device, load_llm, load_nllb
from acoli_nmt.translator import Translator


class Trainer:
    """
    Full training pipeline.

        cfg = Config(epochs=3, lr=1e-4)
        t = Trainer(cfg)
        t.load_data()
        t.train()
        metrics = t.evaluate()
        t.save("./my-model")
        translator = t.get_translator()
    """

    def __init__(self, cfg: Optional[Config] = None):
        self.cfg = cfg or Config()
        self.device = get_device(self.cfg)
        self.model = None
        self.tokenizer = None
        self.splits: Optional[DatasetDict] = None
        self.tokenized: Optional[DatasetDict] = None
        self.hf_trainer = None
        print(f"Config: {self.cfg.summary()}")
        print(f"Device: {self.device}")

    # ── Data ────────────────────────────────────────────────────────────

    def load_data(self) -> DatasetDict:
        """Load all sources, clean, dedup, split."""
        parts = load_all_sources(self.cfg)
        self.splits = clean_and_split(parts, seed=self.cfg.seed)
        return self.splits

    # ── Model ───────────────────────────────────────────────────────────

    def load_model(self):
        """Load tokenizer + model + LoRA."""
        if self.cfg.model_path == "nllb":
            self.tokenizer, self.model = load_nllb(self.cfg, self.device)
        else:
            self.tokenizer, self.model = load_llm(self.cfg, self.device)

        if self.cfg.use_lora:
            self.model = apply_lora(self.model, self.cfg)

    # ── Tokenization ────────────────────────────────────────────────────

    def tokenize(self):
        """Tokenize splits for training."""
        assert self.splits is not None, "Call load_data() first"
        if self.model is None:
            self.load_model()

        cfg = self.cfg

        if cfg.model_path == "nllb":
            tok = self.tokenizer

            def preprocess(examples):
                all_ids, all_attn, all_labs = [], [], []
                for en, ach in zip(examples["en"], examples["ach"]):
                    # en → ach
                    tok.src_lang = cfg.src_lang
                    s = tok(en, max_length=cfg.max_len, truncation=True, padding="max_length")
                    tok.src_lang = cfg.tgt_lang
                    t = tok(ach, max_length=cfg.max_len, truncation=True, padding="max_length")
                    tok.src_lang = cfg.src_lang
                    labs = [(x if x != tok.pad_token_id else -100) for x in t["input_ids"]]
                    all_ids.append(s["input_ids"])
                    all_attn.append(s["attention_mask"])
                    all_labs.append(labs)

                    if cfg.nllb_bidirectional:
                        # ach → en
                        tok.src_lang = cfg.tgt_lang
                        s2 = tok(ach, max_length=cfg.max_len, truncation=True, padding="max_length")
                        tok.src_lang = cfg.src_lang
                        t2 = tok(en, max_length=cfg.max_len, truncation=True, padding="max_length")
                        labs2 = [(x if x != tok.pad_token_id else -100) for x in t2["input_ids"]]
                        all_ids.append(s2["input_ids"])
                        all_attn.append(s2["attention_mask"])
                        all_labs.append(labs2)
                        tok.src_lang = cfg.src_lang

                return {"input_ids": all_ids, "attention_mask": all_attn, "labels": all_labs}

            print(f"⏳ Tokenizing (NLLB, bidir={cfg.nllb_bidirectional}) …")
            self.tokenized = self.splits.map(
                preprocess, batched=True, batch_size=512,
                remove_columns=["en", "ach"], desc="Tok",
            )

        else:  # LLM
            FEW_EN2ACH = (
                "Translate English to Acholi.\n\n"
                "English: Good morning, how are you?\nAcholi: Oyot, i nongo nining?\n\n"
                "English: The water is clean.\nAcholi: Pii tye maler.\n\n"
                "English: {en}\nAcholi:"
            )
            FEW_ACH2EN = (
                "Translate Acholi to English.\n\n"
                "Acholi: Oyot, i nongo nining?\nEnglish: Good morning, how are you?\n\n"
                "Acholi: Pii tye maler.\nEnglish: The water is clean.\n\n"
                "Acholi: {ach}\nEnglish:"
            )

            def build_pairs(examples):
                p, c = [], []
                for en, ach in zip(examples["en"], examples["ach"]):
                    p.append(FEW_EN2ACH.format(en=en)); c.append(f" {ach}")
                    p.append(FEW_ACH2EN.format(ach=ach)); c.append(f" {en}")
                return {"prompt": p, "completion": c}

            def tok_llm(examples):
                ids, labs, att = [], [], []
                tok = self.tokenizer
                for pr, co in zip(examples["prompt"], examples["completion"]):
                    full = pr + co + tok.eos_token
                    enc = tok(full, max_length=cfg.max_len * 3, truncation=True, padding="max_length")
                    pl = len(tok(pr, add_special_tokens=False)["input_ids"])
                    lb = list(enc["input_ids"])
                    for i in range(min(pl, len(lb))):
                        lb[i] = -100
                    lb = [(x if x != tok.pad_token_id else -100) for x in lb]
                    ids.append(enc["input_ids"]); labs.append(lb); att.append(enc["attention_mask"])
                return {"input_ids": ids, "attention_mask": att, "labels": labs}

            print("⏳ Tokenizing (LLM) …")
            inst = self.splits.map(build_pairs, batched=True, batch_size=1024,
                                   remove_columns=["en", "ach"], desc="Inst")
            self.tokenized = inst.map(tok_llm, batched=True, batch_size=512,
                                      remove_columns=["prompt", "completion"], desc="Tok")

        self.tokenized.set_format("torch")
        for k, v in self.tokenized.items():
            print(f"  {k:12s}: {v.num_rows:>7,}")

    # ── Training ────────────────────────────────────────────────────────

    def train(self):
        """Run the full training loop."""
        if self.tokenized is None:
            self.tokenize()

        cfg = self.cfg
        import evaluate as hf_evaluate

        bleu_m = hf_evaluate.load("sacrebleu")
        chrf_m = hf_evaluate.load("chrf")

        if cfg.model_path == "nllb":
            from transformers import (
                DataCollatorForSeq2Seq,
                EarlyStoppingCallback,
                Seq2SeqTrainer,
                Seq2SeqTrainingArguments,
            )

            def compute_metrics(eval_pred):
                preds, labels = eval_pred
                if isinstance(preds, tuple):
                    preds = preds[0]
                preds = np.where((preds >= 0) & (preds < len(self.tokenizer)), preds, self.tokenizer.pad_token_id)
                labels = np.where(labels != -100, labels, self.tokenizer.pad_token_id)
                dp = [p.strip() for p in self.tokenizer.batch_decode(preds, skip_special_tokens=True)]
                dl = [[l.strip()] for l in self.tokenizer.batch_decode(labels, skip_special_tokens=True)]
                return {
                    "bleu": round(bleu_m.compute(predictions=dp, references=dl)["score"], 2),
                    "chrf": round(chrf_m.compute(predictions=dp, references=dl)["score"], 2),
                }

            args = Seq2SeqTrainingArguments(
                output_dir=cfg.output_dir,
                num_train_epochs=cfg.epochs,
                per_device_train_batch_size=cfg.batch_size,
                per_device_eval_batch_size=cfg.batch_size,
                gradient_accumulation_steps=cfg.grad_accum,
                learning_rate=cfg.lr,
                lr_scheduler_type="cosine", warmup_ratio=cfg.warmup, weight_decay=0.01,
                eval_strategy="steps", eval_steps=500,
                save_strategy="steps", save_steps=500, save_total_limit=3,
                load_best_model_at_end=True,
                metric_for_best_model="chrf", greater_is_better=True,
                predict_with_generate=True, generation_max_length=cfg.max_len,
                fp16=self.device.type == "cuda",
                gradient_checkpointing=True,
                dataloader_num_workers=2, logging_steps=50, report_to="none",
            )
            coll = DataCollatorForSeq2Seq(
                self.tokenizer, model=self.model, padding=True, label_pad_token_id=-100,
            )
            self.hf_trainer = Seq2SeqTrainer(
                model=self.model, args=args,
                train_dataset=self.tokenized["train"],
                eval_dataset=self.tokenized["validation"],
                processing_class=self.tokenizer, data_collator=coll,
                compute_metrics=compute_metrics,
                callbacks=[EarlyStoppingCallback(early_stopping_patience=3)],
            )
        else:
            from transformers import (
                DataCollatorForLanguageModeling,
                EarlyStoppingCallback,
                Trainer as HFTrainer,
                TrainingArguments,
            )

            args = TrainingArguments(
                output_dir=cfg.output_dir,
                num_train_epochs=cfg.epochs,
                per_device_train_batch_size=cfg.batch_size // 2,
                per_device_eval_batch_size=cfg.batch_size // 2,
                gradient_accumulation_steps=cfg.grad_accum * 2,
                learning_rate=cfg.lr,
                lr_scheduler_type="cosine", warmup_ratio=cfg.warmup, weight_decay=0.01,
                eval_strategy="steps", eval_steps=500,
                save_strategy="steps", save_steps=500, save_total_limit=3,
                load_best_model_at_end=True,
                metric_for_best_model="eval_loss", greater_is_better=False,
                fp16=self.device.type == "cuda",
                gradient_checkpointing=True,
                dataloader_num_workers=2, logging_steps=50, report_to="none",
            )
            coll = DataCollatorForLanguageModeling(self.tokenizer, mlm=False)
            self.hf_trainer = HFTrainer(
                model=self.model, args=args,
                train_dataset=self.tokenized["train"],
                eval_dataset=self.tokenized["validation"],
                processing_class=self.tokenizer, data_collator=coll,
                callbacks=[EarlyStoppingCallback(early_stopping_patience=3)],
            )

        print(f"🚀 Training ({cfg.model_path.upper()}) …")
        result = self.hf_trainer.train()
        print(f"✅ Done — loss: {result.training_loss:.4f}")
        return result

    # ── Evaluation ──────────────────────────────────────────────────────

    def evaluate(self) -> dict[str, dict[str, float]]:
        """Bidirectional evaluation on test set.  Returns {direction: metrics}."""
        assert self.splits is not None
        self.model.eval()
        translator = self.get_translator()
        td = self.splits["test"]

        en_src = [r["en"] for r in td]
        ach_ref = [r["ach"] for r in td]
        ach_hyp = translator.translate_batch(en_src, "en2ach")

        print("\n📊 EN → ACH")
        m1 = score_all(ach_hyp, ach_ref, en_src)
        for k, v in m1.items():
            print(f"   {k:5s}: {v}")

        ach_src = [r["ach"] for r in td]
        en_ref = [r["en"] for r in td]
        en_hyp = translator.translate_batch(ach_src, "ach2en")

        print("\n📊 ACH → EN")
        m2 = score_all(en_hyp, en_ref, ach_src)
        for k, v in m2.items():
            print(f"   {k:5s}: {v}")

        # Samples
        for lab, ss, rr, hh in [
            ("EN→ACH", en_src, ach_ref, ach_hyp),
            ("ACH→EN", ach_src, en_ref, en_hyp),
        ]:
            print(f"\n🔍 {lab}:")
            for i in range(min(5, len(hh))):
                print(f"  SRC: {ss[i]}\n  REF: {rr[i]}\n  HYP: {hh[i]}\n")

        return {"en2ach": m1, "ach2en": m2}

    # ── Save / Load ─────────────────────────────────────────────────────

    def save(self, path: Optional[str] = None):
        """Save model + tokenizer."""
        p = path or self.cfg.save_dir
        print(f"💾 Saving → {p}")
        if self.cfg.use_lora:
            self.model.save_pretrained(p)
        elif self.hf_trainer:
            self.hf_trainer.save_model(p)
        else:
            self.model.save_pretrained(p)
        self.tokenizer.save_pretrained(p)

    def get_translator(self) -> Translator:
        """Get a Translator instance from the current model."""
        return Translator(
            model=self.model,
            tokenizer=self.tokenizer,
            model_path=self.cfg.model_path,
            src_lang=self.cfg.src_lang,
            tgt_lang=self.cfg.tgt_lang,
            max_len=self.cfg.max_len,
            device=self.device,
        )
