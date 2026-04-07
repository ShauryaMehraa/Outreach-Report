"""
Whisper ASR wrapper.

Loads a multilingual Whisper model and transcribes diarized audio chunks into
native-language text segments for the downstream translation stage.
"""

from __future__ import annotations

import os
from typing import Any, Dict

import torch
from transformers import WhisperForConditionalGeneration, WhisperProcessor

MODEL_ID = os.getenv("WHISPER_MODEL_ID", "openai/whisper-small")

WHISPER_LANGUAGE_MAP = {
    "pa": "punjabi",
    "punjabi": "punjabi",
    "hi": "hindi",
    "hindi": "hindi",
    "ta": "tamil",
    "tamil": "tamil",
    "te": "telugu",
    "telugu": "telugu",
    "mr": "marathi",
    "marathi": "marathi",
    "kn": "kannada",
    "kannada": "kannada",
    "gu": "gujarati",
    "gujarati": "gujarati",
    "bn": "bengali",
    "bengali": "bengali",
    "or": "odia",
    "odia": "odia",
    "ml": "malayalam",
    "malayalam": "malayalam",
}


# =============================================================================
# MODEL LOADER
# =============================================================================

def load_asr_model(device: str) -> Dict[str, Any]:
    print(f"Loading Whisper ASR model: {MODEL_ID}...")
    processor = WhisperProcessor.from_pretrained(MODEL_ID)
    model = WhisperForConditionalGeneration.from_pretrained(
        MODEL_ID,
        torch_dtype=torch.float16 if device == "cuda" else torch.float32,
        low_cpu_mem_usage=True,
    ).to(device)
    model.eval()

    if hasattr(model.config, "forced_decoder_ids"):
        model.config.forced_decoder_ids = None

    return {
        "processor": processor,
        "model": model,
    }


# =============================================================================
# INFERENCE
# =============================================================================

def transcribe_chunk(
    model_bundle: Dict[str, Any],
    chunk: torch.Tensor,
    language: str,
    device: str,
    decoder: str = "ctc",
) -> str:
    """
    Run ASR on a single audio chunk (already on the correct device).
    Returns the transcribed string, or empty string on failure.
    """
    try:
        processor: WhisperProcessor = model_bundle["processor"]
        model: WhisperForConditionalGeneration = model_bundle["model"]
        whisper_language = WHISPER_LANGUAGE_MAP.get(str(language).strip().lower())

        audio = chunk.detach().float().cpu().squeeze().numpy()
        inputs = processor(audio, sampling_rate=16000, return_tensors="pt")
        input_features = inputs.input_features.to(device)

        generate_kwargs: Dict[str, Any] = {}
        if whisper_language:
            generate_kwargs["language"] = whisper_language
            generate_kwargs["forced_decoder_ids"] = processor.get_decoder_prompt_ids(
                language=whisper_language,
                task="transcribe",
            )

        with torch.no_grad():
            generated_ids = model.generate(
                input_features,
                **generate_kwargs,
            )

        text = processor.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
        return text.strip() if text else ""
    except Exception as e:
        print(f"    ASR failed: {e}")
        return ""