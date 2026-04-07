"""
Thin wrapper around pyannote/speaker-diarization-3.1.
Returns a list of (start, end, speaker_id) turn tuples.
Requires HF_TOKEN in .env — the pyannote model is gated on HuggingFace.
"""
import os
import torch
import soundfile as sf
from pyannote.audio import Pipeline

# =============================================================================
# MODEL LOADER
# =============================================================================

def load_diarization_pipeline(device: str) -> Pipeline:
    token = os.getenv("HF_TOKEN")
    if not token:
        raise EnvironmentError(
            "HF_TOKEN not found. Add it to your .env file:\n"
            "  HF_TOKEN=hf_xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx\n"
            "Also accept the model license at: "
            "https://huggingface.co/pyannote/speaker-diarization-3.1"
        )
    print("Loading diarization pipeline...")
    pipeline = Pipeline.from_pretrained(
        "pyannote/speaker-diarization-3.1",
        token=token,
    )
    pipeline.to(torch.device(device))
    return pipeline

# =============================================================================
# INFERENCE
# =============================================================================

def diarize(pipeline: Pipeline, audio_path: str) -> list[tuple[float, float, str]]:
    """
    Run full-audio diarization and return a sorted list of
    (start_sec, end_sec, speaker_id) tuples.
    Loads audio manually to avoid torchcodec issues.
    """
    print("Loading audio for diarization...")
    waveform, sample_rate = sf.read(audio_path, dtype="float32")
    if waveform.ndim == 1:
        waveform = waveform[None, :]  # (1, samples)
    else:
        waveform = waveform.T         # (channels, samples)
    waveform_tensor = torch.tensor(waveform)

    audio_input = {
        "waveform": waveform_tensor,
        "sample_rate": sample_rate,
    }

    print("Running speaker diarization...")
    diarization = pipeline(audio_input)

    turns = [
        (turn.start, turn.end, speaker)
        for turn, _, speaker in diarization.speaker_diarization.itertracks(yield_label=True)
    ]
    print(f"Found {len(turns)} speaker turns.")
    return turns
