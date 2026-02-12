"""High-level Translator for inference after training."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import torch
from tqdm.auto import tqdm


class Translator:
    """
    Load a trained Acoli ↔ English model and translate.

        t = Translator.from_pretrained("./acoli-en-nllb-final")
        t.en_to_ach("How are you?")
        t.ach_to_en("Itye nining?")
        t.translate("Hello world", direction="en2ach")
        t.translate_batch(["Hello", "Goodbye"], direction="en2ach")
    """

    def __init__(
        self,
        model,
        tokenizer,
        model_path: str = "nllb",
        src_lang: str = "eng_Latn",
        tgt_lang: str = "ach_Latn",
        max_len: int = 256,
        device: Optional[torch.device] = None,
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.model_path = model_path
        self.src_lang = src_lang
        self.tgt_lang = tgt_lang
        self.max_len = max_len
        self.device = device or next(model.parameters()).device
        self.model.eval()

    # ── Factory ─────────────────────────────────────────────────────────

    @classmethod
    def from_pretrained(
        cls,
        path: str,
        model_path: str = "nllb",
        base_model: Optional[str] = None,
        src_lang: str = "eng_Latn",
        tgt_lang: str = "ach_Latn",
        related_lang: str = "luo_Latn",
        max_len: int = 256,
        device: Optional[str] = None,
    ) -> "Translator":
        """
        Load a saved model (LoRA adapters or full checkpoint).

        Args:
            path: Directory containing saved model/adapter + tokenizer.
            model_path: "nllb" or "llm".
            base_model: Base model name (only needed for LoRA adapters).
            device: "cuda", "cpu", or None for auto.
        """
        dev = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        p = Path(path)

        if model_path == "nllb":
            from transformers import NllbTokenizer, AutoModelForSeq2SeqLM
            from transformers.models.nllb.tokenization_nllb import FAIRSEQ_LANGUAGE_CODES

            # Check if this is a LoRA adapter or full model
            is_lora = (p / "adapter_config.json").exists()

            # Load tokenizer (saved with extended language_codes)
            tokenizer = NllbTokenizer.from_pretrained(str(p))

            if is_lora:
                from peft import PeftModel

                base = base_model or "facebook/nllb-200-distilled-600M"
                base_tok = NllbTokenizer.from_pretrained(str(p))
                model = AutoModelForSeq2SeqLM.from_pretrained(
                    base,
                    torch_dtype=torch.float16 if dev.type == "cuda" else torch.float32,
                ).to(dev)
                model.resize_token_embeddings(len(base_tok))
                model = PeftModel.from_pretrained(model, str(p)).to(dev)
                model = model.merge_and_unload()
            else:
                model = AutoModelForSeq2SeqLM.from_pretrained(
                    str(p),
                    torch_dtype=torch.float16 if dev.type == "cuda" else torch.float32,
                ).to(dev)

        elif model_path == "llm":
            from transformers import AutoTokenizer, AutoModelForCausalLM

            is_lora = (p / "adapter_config.json").exists()
            tokenizer = AutoTokenizer.from_pretrained(str(p))
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token

            if is_lora:
                from peft import PeftModel

                base = base_model or "Unbabel/TowerInstruct-7B-v0.2"
                model = AutoModelForCausalLM.from_pretrained(
                    base, torch_dtype=torch.float16, device_map="auto",
                )
                model = PeftModel.from_pretrained(model, str(p))
                model = model.merge_and_unload()
            else:
                model = AutoModelForCausalLM.from_pretrained(
                    str(p), torch_dtype=torch.float16, device_map="auto",
                )
        else:
            raise ValueError(f"Unknown model_path: {model_path!r}")

        return cls(
            model=model,
            tokenizer=tokenizer,
            model_path=model_path,
            src_lang=src_lang,
            tgt_lang=tgt_lang,
            max_len=max_len,
            device=dev,
        )

    # ── Translation core ────────────────────────────────────────────────

    @torch.no_grad()
    def translate(
        self,
        text: str,
        direction: str = "en2ach",
        num_beams: int = 4,
    ) -> str:
        """
        Translate a single string.

        Args:
            text: Input text.
            direction: "en2ach" or "ach2en".
            num_beams: Beam search width.
        """
        if not text.strip():
            return ""

        if self.model_path == "nllb":
            if direction == "en2ach":
                self.tokenizer.src_lang = self.src_lang
                forced = self.tokenizer.convert_tokens_to_ids(self.tgt_lang)
            else:
                self.tokenizer.src_lang = self.tgt_lang
                forced = self.tokenizer.convert_tokens_to_ids(self.src_lang)

            enc = self.tokenizer(
                text, return_tensors="pt", max_length=self.max_len, truncation=True,
            ).to(self.device)
            out = self.model.generate(
                **enc,
                forced_bos_token_id=forced,
                max_new_tokens=self.max_len,
                num_beams=num_beams,
            )
            return self.tokenizer.decode(out[0], skip_special_tokens=True).strip()

        else:  # LLM
            if direction == "en2ach":
                prompt = (
                    "Translate English to Acholi.\n\n"
                    "English: Good morning, how are you?\nAcholi: Oyot, i nongo nining?\n\n"
                    "English: The water is clean.\nAcholi: Pii tye maler.\n\n"
                    f"English: {text}\nAcholi:"
                )
            else:
                prompt = (
                    "Translate Acholi to English.\n\n"
                    "Acholi: Oyot, i nongo nining?\nEnglish: Good morning, how are you?\n\n"
                    "Acholi: Pii tye maler.\nEnglish: The water is clean.\n\n"
                    f"Acholi: {text}\nEnglish:"
                )
            enc = self.tokenizer(
                prompt, return_tensors="pt", truncation=True, max_length=self.max_len * 2,
            ).to(self.device)
            out = self.model.generate(
                **enc, max_new_tokens=self.max_len, num_beams=num_beams, do_sample=False,
            )
            decoded = self.tokenizer.decode(
                out[0][enc["input_ids"].shape[1]:], skip_special_tokens=True,
            ).strip()
            return decoded.split("\n")[0].strip()

    # ── Convenience methods ─────────────────────────────────────────────

    def en_to_ach(self, text: str, **kw) -> str:
        return self.translate(text, direction="en2ach", **kw)

    def ach_to_en(self, text: str, **kw) -> str:
        return self.translate(text, direction="ach2en", **kw)

    def translate_batch(
        self,
        texts: list[str],
        direction: str = "en2ach",
        num_beams: int = 4,
        show_progress: bool = True,
    ) -> list[str]:
        """Translate a list of strings."""
        it = tqdm(texts, desc=direction) if show_progress else texts
        return [self.translate(t, direction, num_beams) for t in it]
