"""Model and tokenizer loading with proper ach_Latn registration."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from acoli_nmt.config import Config


def get_device(cfg: Config) -> torch.device:
    if cfg.device:
        return torch.device(cfg.device)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_nllb(cfg: Config, device: torch.device):
    """
    Load NLLB tokenizer (with extended language codes) and model.

    Adds ach_Latn to FAIRSEQ_LANGUAGE_CODES via additional_special_tokens
    (transformers 5.x) or extra_special_tokens parameter.
    """
    from transformers import NllbTokenizer, AutoModelForSeq2SeqLM
    from transformers.models.nllb.tokenization_nllb import FAIRSEQ_LANGUAGE_CODES

    print(f"⏳ Loading NLLB: {cfg.nllb_model}")

    # ── Extend language codes ───────────────────────────────────────────
    new_codes = list(FAIRSEQ_LANGUAGE_CODES)
    if cfg.tgt_lang not in new_codes:
        new_codes.append(cfg.tgt_lang)
        print(f"   Adding {cfg.tgt_lang} to language codes")

    # transformers 5.x uses additional_special_tokens / extra_special_tokens
    # (the old language_codes kwarg no longer exists)
    tokenizer = NllbTokenizer.from_pretrained(
        cfg.nllb_model,
        additional_special_tokens=new_codes,
    )

    # Verify the new token was registered
    ach_id = tokenizer.convert_tokens_to_ids(cfg.tgt_lang)
    if ach_id == tokenizer.unk_token_id:
        # Fallback: manually add as special token and retry
        print(f"   ⚠️ {cfg.tgt_lang} not in vocab, adding manually …")
        tokenizer.add_special_tokens(
            {"additional_special_tokens": list(set(tokenizer.additional_special_tokens + [cfg.tgt_lang]))}
        )
        ach_id = tokenizer.convert_tokens_to_ids(cfg.tgt_lang)
    assert ach_id != tokenizer.unk_token_id, f"{cfg.tgt_lang} still mapped to <unk>!"
    print(f"   ✓ {cfg.tgt_lang} → id {ach_id}")
    print(f"   ✓ {cfg.src_lang} → id {tokenizer.convert_tokens_to_ids(cfg.src_lang)}")

    # ── Load model with Flash Attention 2 fallback ──────────────────────
    attn_impl = "eager"
    if device.type == "cuda":
        try:
            model = AutoModelForSeq2SeqLM.from_pretrained(
                cfg.nllb_model,
                torch_dtype=torch.float16,
                attn_implementation="flash_attention_2",
            ).to(device)
            attn_impl = "flash_attention_2"
        except Exception as e:
            print(f"   ⚠️ FA2 unavailable ({e}); using eager")
            model = AutoModelForSeq2SeqLM.from_pretrained(
                cfg.nllb_model, torch_dtype=torch.float16,
            ).to(device)
    else:
        model = AutoModelForSeq2SeqLM.from_pretrained(cfg.nllb_model).to(device)
    print(f"   Attention: {attn_impl}")

    # ── Resize embeddings + init from related language ──────────────────
    old_size = model.model.shared.weight.shape[0]
    model.resize_token_embeddings(len(tokenizer))
    new_size = model.model.shared.weight.shape[0]

    if new_size > old_size:
        related_id = tokenizer.convert_tokens_to_ids(cfg.related_lang)
        if related_id != tokenizer.unk_token_id:
            with torch.no_grad():
                model.model.shared.weight.data[ach_id] = (
                    model.model.shared.weight.data[related_id].clone()
                )
                if (
                    hasattr(model, "lm_head")
                    and model.lm_head.weight is not model.model.shared.weight
                ):
                    model.lm_head.weight.data[ach_id] = (
                        model.lm_head.weight.data[related_id].clone()
                    )
            print(f"   ✓ Embedding init from {cfg.related_lang}")
        else:
            print(f"   ⚠️ {cfg.related_lang} not found; random init")

    model.gradient_checkpointing_enable()
    print(f"   ✓ Gradient checkpointing | Params: {sum(p.numel() for p in model.parameters()):,}")
    return tokenizer, model


def load_llm(cfg: Config, device: torch.device):
    """Load Tower-7B with QLoRA 4-bit quantization."""
    from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

    print(f"⏳ Loading LLM (4-bit QLoRA): {cfg.llm_model}")
    bnb = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
    )
    tokenizer = AutoTokenizer.from_pretrained(cfg.llm_model, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    try:
        model = AutoModelForCausalLM.from_pretrained(
            cfg.llm_model,
            quantization_config=bnb,
            device_map="auto",
            attn_implementation="flash_attention_2",
        )
        print("   Attention: flash_attention_2")
    except Exception:
        model = AutoModelForCausalLM.from_pretrained(
            cfg.llm_model, quantization_config=bnb, device_map="auto",
        )
        print("   Attention: eager")

    model.gradient_checkpointing_enable()
    print(f"   Params: {sum(p.numel() for p in model.parameters()):,}")
    return tokenizer, model


def apply_lora(model, cfg: Config):
    """Wrap model with LoRA adapters."""
    from peft import get_peft_model, LoraConfig, TaskType, prepare_model_for_kbit_training

    if cfg.model_path == "nllb":
        task = TaskType.SEQ_2_SEQ_LM
        targets = ["q_proj", "v_proj"]
    else:
        task = TaskType.CAUSAL_LM
        targets = ["q_proj", "k_proj", "v_proj", "o_proj"]
        if getattr(model, "is_loaded_in_4bit", False) or getattr(
            model, "is_loaded_in_8bit", False
        ):
            model = prepare_model_for_kbit_training(model)

    model = get_peft_model(
        model,
        LoraConfig(
            task_type=task,
            r=cfg.lora_r,
            lora_alpha=cfg.lora_alpha,
            lora_dropout=cfg.lora_dropout,
            target_modules=targets,
        ),
    )
    model.print_trainable_parameters()
    return model
