"""
Acoli ↔ English Neural Machine Translation Library (v0.6)

Usage:
    from acoli_nmt import Translator, Trainer, Config

    # Quick translation with a trained model
    t = Translator.from_pretrained("./acoli-en-nllb-final")
    print(t.en_to_ach("How are you today?"))
    print(t.ach_to_en("Itye nining?"))

    # Full training pipeline
    cfg = Config()
    trainer = Trainer(cfg)
    trainer.load_data()
    trainer.train()
    trainer.evaluate()
    trainer.save("./my-model")
"""

__version__ = "0.6.0"

from acoli_nmt.config import Config
from acoli_nmt.translator import Translator
from acoli_nmt.trainer import Trainer

__all__ = ["Config", "Translator", "Trainer", "__version__"]
