"""
pipeline/translation/indictrans2.py

Translates native-language transcript entries to English using
AI4Bharat IndicTrans2 (ai4bharat/indictrans2-indic-en-1B).

Why IndicTrans2?
---------------
- Free and open-source (Apache 2.0, AI4Bharat). No API key or paid subscription.
- Purpose-built for 22 Indian languages → English; consistently outperforms
  general-purpose models (Sarvam, NLLB, Helsinki-NLP) on Indic text.
- Seq2seq architecture (encoder-decoder) is better suited for translation than
  causal LMs like sarvam-translate.
- Only ~3 GB VRAM in bfloat16 (vs ~7 GB for sarvam-translate).
- Public model — HuggingFace token is NOT required.

Interface (drop-in replacement for sarvam_translate.py):
  IndicTrans2Translator.translate_transcript()  ← same signature
  IndicTrans2Translator.translate_batch()       ← same signature

FLORES-200 source codes (e.g. "pan_Guru", "hin_Deva") are used directly —
no remapping needed because IndicTrans2 uses the same convention.
Target is always "eng_Latn" (English).

Installation:
  pip install IndicTransToolkit
  (IndicTrans2 model weights download automatically on first run via HuggingFace)
"""

from __future__ import annotations

import logging

import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

log = logging.getLogger(__name__)

MODEL_ID = "ai4bharat/indictrans2-indic-en-1B"

# ---------------------------------------------------------------------------
# Optional: IndicTransToolkit provides the best pre/post processing pipeline
# (Unicode normalisation, Indic numeral handling, script-specific tokenisation).
# The translator works without it but quality is marginally better with it.
# ---------------------------------------------------------------------------
try:
    from IndicTransToolkit import IndicProcessor as _IndicProcessor
    _HAS_TOOLKIT = True
    log.info("IndicTransToolkit found — using full pre/post processing.")
except ImportError:
    _HAS_TOOLKIT = False
    log.warning(
        "IndicTransToolkit not installed. "
        "Install it for best translation quality:\n"
        "  pip install IndicTransToolkit\n"
        "Continuing with raw tokenizer fallback."
    )


# =============================================================================
# TRANSLATOR
# =============================================================================

class IndicTrans2Translator:
    """
    Indic → English translator backed by ai4bharat/indictrans2-indic-en-1B.

    Free, open-source, and specifically optimised for all 10 Indic languages
    supported by this pipeline. Uses ~3 GB VRAM in bfloat16.

    Drop-in replacement for SarvamTranslator — identical public interface.
    """

    def __init__(self, device: str = "cuda"):
        self.device = device

        log.info(f"Loading IndicTrans2 ({MODEL_ID}) on {device} ...")
        self.tokenizer = AutoTokenizer.from_pretrained(
            MODEL_ID,
            trust_remote_code=True,
        )
        self.model = AutoModelForSeq2SeqLM.from_pretrained(
            MODEL_ID,
            trust_remote_code=True,
            torch_dtype=torch.bfloat16 if device == "cuda" else torch.float32,
        ).to(device)
        self.model.eval()

        # Instantiate IndicProcessor once — reused for every batch
        self._processor = _IndicProcessor(inference=True) if _HAS_TOOLKIT else None

        if torch.cuda.is_available():
            log.info(
                f"IndicTrans2 loaded. "
                f"VRAM allocated: {torch.cuda.memory_allocated() / 1e9:.2f} GB"
            )
        else:
            log.info("IndicTrans2 loaded on CPU.")

    # ------------------------------------------------------------------
    # CORE BATCH TRANSLATION
    # ------------------------------------------------------------------

    def translate_batch(
        self,
        texts: list[str],
        src_lang: str,
        tgt_lang: str = "eng_Latn",
    ) -> list[str]:
        """
        Translate a list of strings from src_lang to tgt_lang.

        Args:
            texts:    Source strings.
            src_lang: FLORES-200 source language code (e.g. "pan_Guru").
            tgt_lang: FLORES-200 target language code (default: "eng_Latn").

        Returns:
            Translated strings — same length as input; empty strings preserved.
        """
        # Track positions of non-empty inputs so empties pass through unchanged
        indices: list[int] = []
        clean: list[str] = []
        for i, t in enumerate(texts):
            if t and t.strip():
                indices.append(i)
                clean.append(t.strip())

        if not clean:
            return [""] * len(texts)

        # Pre-processing ─────────────────────────────────────────────────────
        if self._processor is not None:
            preprocessed = self._processor.preprocess_batch(
                clean, src_lang=src_lang, tgt_lang=tgt_lang
            )
        else:
            preprocessed = clean

        # Tokenise ───────────────────────────────────────────────────────────
        inputs = self.tokenizer(
            preprocessed,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=256,
        ).to(self.device)

        # Inference ──────────────────────────────────────────────────────────
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                num_beams=5,
                num_return_sequences=1,
                max_length=256,
                forced_bos_token_id=self.tokenizer.convert_tokens_to_ids(tgt_lang),
            )

        # Decode ─────────────────────────────────────────────────────────────
        decoded = self.tokenizer.batch_decode(outputs, skip_special_tokens=True)

        if self._processor is not None:
            translated = self._processor.postprocess_batch(decoded, lang=tgt_lang)
        else:
            translated = decoded

        # Re-insert empties at their original positions ──────────────────────
        result: list[str] = [""] * len(texts)
        for idx, trans in zip(indices, translated):
            result[idx] = (trans or "").strip()
        return result

    # ------------------------------------------------------------------
    # STRUCTURED TRANSCRIPT TRANSLATION
    # ------------------------------------------------------------------

    def translate_transcript(
        self,
        entries: list[dict],
        src_lang: str,
        tgt_lang: str = "eng_Latn",
        batch_size: int = 32,
    ) -> list[dict]:
        """
        Fill the 'translated_text' field of each entry in-place.
        Reads entry['original_text'], writes entry['translated_text'].
        Returns the same list for chaining.
        """
        total_batches = (len(entries) + batch_size - 1) // batch_size

        for i in range(0, len(entries), batch_size):
            batch = entries[i : i + batch_size]
            texts = [e.get("original_text", "") for e in batch]

            log.info(
                f"  Translating batch {i // batch_size + 1}/{total_batches} "
                f"({len(batch)} segments) ..."
            )
            translated_texts = self.translate_batch(texts, src_lang, tgt_lang)

            for entry, translated in zip(batch, translated_texts):
                entry["translated_text"] = translated

        log.info(f"Translation complete. {len(entries)} segments translated.")
        return entries