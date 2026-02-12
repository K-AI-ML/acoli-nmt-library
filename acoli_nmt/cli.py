"""Command-line interface: train, eval, translate, serve."""

from __future__ import annotations

import argparse
import sys


def _base_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(add_help=False)
    p.add_argument("--model-path", default="nllb", choices=["nllb", "llm"])
    p.add_argument("--nllb-model", default="facebook/nllb-200-distilled-600M")
    p.add_argument("--llm-model", default="Unbabel/TowerInstruct-7B-v0.2")
    p.add_argument("--device", default=None)
    p.add_argument("--max-len", type=int, default=256)
    p.add_argument("--seed", type=int, default=42)
    return p


def _cfg_from_args(args) -> "Config":
    from acoli_nmt.config import Config

    return Config(
        model_path=args.model_path,
        nllb_model=args.nllb_model,
        llm_model=args.llm_model,
        max_len=args.max_len,
        seed=args.seed,
        device=args.device,
        **{k: v for k, v in vars(args).items() if hasattr(Config, k) and k not in [
            "model_path", "nllb_model", "llm_model", "max_len", "seed", "device",
        ]},
    )


# ── acoli-train ─────────────────────────────────────────────────────────

def train_cli():
    p = argparse.ArgumentParser(
        description="Train Acoli ↔ English NMT model",
        parents=[_base_parser()],
    )
    p.add_argument("--epochs", type=int, default=5)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--grad-accum", type=int, default=4)
    p.add_argument("--lora-r", type=int, default=16)
    p.add_argument("--no-lora", action="store_true")
    p.add_argument("--no-bidir", action="store_true")
    p.add_argument("--output-dir", default="./acoli-en-nllb-ft")
    p.add_argument("--save-dir", default="./acoli-en-nllb-final")
    args = p.parse_args()

    from acoli_nmt.config import Config
    from acoli_nmt.trainer import Trainer

    cfg = Config(
        model_path=args.model_path,
        nllb_model=args.nllb_model,
        llm_model=args.llm_model,
        max_len=args.max_len,
        seed=args.seed,
        device=args.device,
        epochs=args.epochs,
        lr=args.lr,
        batch_size=args.batch_size,
        grad_accum=args.grad_accum,
        lora_r=args.lora_r,
        use_lora=not args.no_lora,
        nllb_bidirectional=not args.no_bidir,
        output_dir=args.output_dir,
        save_dir=args.save_dir,
    )

    trainer = Trainer(cfg)
    trainer.load_data()
    trainer.train()
    metrics = trainer.evaluate()
    trainer.save()
    print(f"\n🏁 Final metrics: {metrics}")


# ── acoli-eval ──────────────────────────────────────────────────────────

def eval_cli():
    p = argparse.ArgumentParser(
        description="Evaluate a trained Acoli NMT model",
        parents=[_base_parser()],
    )
    p.add_argument("checkpoint", help="Path to saved model/adapter directory")
    p.add_argument("--base-model", default=None, help="Base model (for LoRA)")
    args = p.parse_args()

    from acoli_nmt.data import clean_and_split, load_all_sources
    from acoli_nmt.config import Config
    from acoli_nmt.metrics import score_all
    from acoli_nmt.translator import Translator

    cfg = Config(model_path=args.model_path, max_len=args.max_len, seed=args.seed)
    parts = load_all_sources(cfg)
    splits = clean_and_split(parts, seed=cfg.seed)

    t = Translator.from_pretrained(
        args.checkpoint,
        model_path=args.model_path,
        base_model=args.base_model,
        device=args.device,
    )

    td = splits["test"]
    en_src = [r["en"] for r in td]
    ach_ref = [r["ach"] for r in td]

    print("\n📊 EN → ACH")
    ach_hyp = t.translate_batch(en_src, "en2ach")
    m1 = score_all(ach_hyp, ach_ref, en_src)
    for k, v in m1.items():
        print(f"   {k}: {v}")

    print("\n📊 ACH → EN")
    en_hyp = t.translate_batch([r["ach"] for r in td], "ach2en")
    m2 = score_all(en_hyp, [r["en"] for r in td], [r["ach"] for r in td])
    for k, v in m2.items():
        print(f"   {k}: {v}")


# ── acoli-translate ─────────────────────────────────────────────────────

def translate_cli():
    p = argparse.ArgumentParser(
        description="Translate text with a trained model",
        parents=[_base_parser()],
    )
    p.add_argument("checkpoint", help="Path to saved model")
    p.add_argument("text", nargs="?", help="Text to translate (or pipe via stdin)")
    p.add_argument("-d", "--direction", default="en2ach", choices=["en2ach", "ach2en"])
    p.add_argument("--base-model", default=None)
    p.add_argument("--beams", type=int, default=4)
    args = p.parse_args()

    from acoli_nmt.translator import Translator

    t = Translator.from_pretrained(
        args.checkpoint,
        model_path=args.model_path,
        base_model=args.base_model,
        device=args.device,
    )

    if args.text:
        print(t.translate(args.text, direction=args.direction, num_beams=args.beams))
    else:
        # Read from stdin (pipe mode)
        for line in sys.stdin:
            line = line.strip()
            if line:
                print(t.translate(line, direction=args.direction, num_beams=args.beams))


# ── acoli-serve ─────────────────────────────────────────────────────────

def serve_cli():
    p = argparse.ArgumentParser(
        description="Launch Gradio translation UI",
        parents=[_base_parser()],
    )
    p.add_argument("checkpoint", help="Path to saved model")
    p.add_argument("--base-model", default=None)
    p.add_argument("--port", type=int, default=7860)
    p.add_argument("--share", action="store_true", help="Create public Gradio link")
    args = p.parse_args()

    from acoli_nmt.translator import Translator
    from acoli_nmt.serve import create_app

    t = Translator.from_pretrained(
        args.checkpoint,
        model_path=args.model_path,
        base_model=args.base_model,
        device=args.device,
    )

    app = create_app(t)
    app.launch(server_port=args.port, share=args.share, show_error=True)
