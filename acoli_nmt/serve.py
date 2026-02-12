"""Gradio UI for the Acoli NMT translator."""

from __future__ import annotations

import time
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from acoli_nmt.translator import Translator


def create_app(translator: "Translator"):
    """Build and return a Gradio Blocks app."""
    import gradio as gr
    from acoli_nmt.metrics import sentence_score

    def translate_single(text, direction, reference=""):
        if not text.strip():
            return "", ""
        t0 = time.time()
        d = "en2ach" if direction == "English → Acholi" else "ach2en"
        result = translator.translate(text.strip(), direction=d)
        elapsed = time.time() - t0
        info = f"⏱ {elapsed:.2f}s"
        if reference.strip():
            s = sentence_score(result, reference.strip())
            info += f"  |  BLEU: {s['bleu']}  |  chrF: {s['chrf']}"
        return result, info

    def translate_batch(text, direction):
        if not text.strip():
            return ""
        lines = [l.strip() for l in text.strip().split("\n") if l.strip()]
        d = "en2ach" if direction == "English → Acholi" else "ach2en"
        results = []
        for line in lines:
            trans = translator.translate(line, direction=d)
            results.append(f"{line}\t→\t{trans}")
        return "\n".join(results)

    EN_EXAMPLES = [
        "How are you today?",
        "The children are playing in the field.",
        "Education is the key to a better future.",
        "We need clean water in our village.",
        "Thank you for helping me.",
        "The doctor said you should rest.",
        "Rain is coming tomorrow.",
        "I love my family very much.",
    ]
    ACH_EXAMPLES = [
        "Itye nining?",
        "Lutino gitye ka tuko i bar.",
        "Kwan aye lagony me kwo maber.",
        "Wamito pii maler i gang wa.",
        "Apwoyo pi konyo na.",
    ]

    def run_gallery(direction):
        examples = EN_EXAMPLES if direction == "English → Acholi" else ACH_EXAMPLES
        d = "en2ach" if direction == "English → Acholi" else "ach2en"
        rows = []
        for ex in examples:
            trans = translator.translate(ex, direction=d)
            rows.append(f"**{ex}**\n→ {trans}\n")
        return "\n".join(rows)

    with gr.Blocks(
        title="Acoli ↔ English NMT",
        theme=gr.themes.Soft(primary_hue="teal", secondary_hue="amber"),
    ) as app:

        gr.Markdown("""
        # 🌍 Acoli ↔ English Translation
        """)

        with gr.Tabs():
            with gr.Tab("🔤 Translate"):
                with gr.Row():
                    direction = gr.Radio(
                        ["English → Acholi", "Acholi → English"],
                        value="English → Acholi", label="Direction",
                    )
                with gr.Row():
                    with gr.Column():
                        input_text = gr.Textbox(label="Input", lines=4,
                                                placeholder="Type or paste text…")
                        reference = gr.Textbox(label="Reference (optional)", lines=2,
                                               placeholder="For BLEU/chrF scoring")
                    with gr.Column():
                        output_text = gr.Textbox(label="Translation", lines=4, interactive=False)
                        metrics_text = gr.Textbox(label="Info", lines=1, interactive=False)
                btn = gr.Button("Translate", variant="primary", size="lg")
                btn.click(translate_single,
                          inputs=[input_text, direction, reference],
                          outputs=[output_text, metrics_text])
                gr.Examples(
                    examples=[
                        ["How are you today?", "English → Acholi"],
                        ["Education is the key to a better future.", "English → Acholi"],
                        ["We need clean water in our village.", "English → Acholi"],
                        ["Itye nining?", "Acholi → English"],
                    ],
                    inputs=[input_text, direction],
                )

            with gr.Tab("📋 Batch"):
                gr.Markdown("One sentence per line.")
                batch_dir = gr.Radio(["English → Acholi", "Acholi → English"],
                                     value="English → Acholi", label="Direction")
                batch_in = gr.Textbox(label="Input", lines=8,
                                      placeholder="How are you?\nThank you.\nGood morning.")
                batch_out = gr.Textbox(label="Results", lines=10, interactive=False)
                gr.Button("Translate All", variant="primary").click(
                    translate_batch, inputs=[batch_in, batch_dir], outputs=[batch_out])

            with gr.Tab("🎯 Gallery"):
                gallery_dir = gr.Radio(["English → Acholi", "Acholi → English"],
                                       value="English → Acholi", label="Direction")
                gallery_out = gr.Markdown()
                gr.Button("Run Examples", variant="secondary").click(
                    run_gallery, inputs=[gallery_dir], outputs=[gallery_out])

    return app
