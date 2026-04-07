"""
pipeline/asr/whisper_asr.py

Drop-in replacement for indic_conformer.py using OpenAI Whisper.
Supports all Indic languages that Whisper covers.
"""

import logging
import numpy as np
import torch
import whisper  # pip install openai-whisper

log = logging.getLogger(__name__)

# Map your project's language codes to Whisper language names
WHISPER_LANG_MAP = {
    "pa": "punjabi",
    "hi": "hindi",
    "ta": "tamil",
    "te": "telugu",
    "mr": "marathi",
    "kn": "kannada",
    "gu": "gujarati",
    "bn": "bengali",
    "or": "odia",
    "ml": "malayalam",
}


def load_asr_model(device: str):
    """
    Load Whisper model. Returns a whisper model object.
    Model sizes: tiny, base, small, medium, large, large-v2, large-v3
    Use 'large-v3' for best Indic language accuracy.
    """
    model_size = "large-v3"  # Change to "medium" if GPU memory is limited
    log.info(f"Loading Whisper {model_size} on {device}...")
    model = whisper.load_model(model_size, device=device)
    log.info("Whisper model loaded.")
    return model


def transcribe_segment(
    asr_model,
    audio_array: np.ndarray,
    sample_rate: int,
    language: str,
    device: str,
) -> str:
    """
    Transcribe a single audio segment (numpy array) using Whisper.

    Args:
        asr_model: Loaded Whisper model
        audio_array: Audio as float32 numpy array
        sample_rate: Must be 16000 for Whisper
        language: Your project's language code (e.g., "hi", "pa")
        device: "cuda" or "cpu"

    Returns:
        Transcribed text string
    """
    whisper_lang = WHISPER_LANG_MAP.get(language, "hindi")

    # Whisper expects 16kHz mono float32
    if sample_rate != 16000:
        import torchaudio
        waveform = torch.tensor(audio_array).unsqueeze(0)
        resampler = torchaudio.transforms.Resample(sample_rate, 16000)
        audio_array = resampler(waveform).squeeze(0).numpy()

    result = asr_model.transcribe(
        audio_array,
        language=whisper_lang,
        task="transcribe",  # Keep original language; use "translate" to get English directly
        fp16=(device == "cuda"),
    )
    return result["text"].strip()