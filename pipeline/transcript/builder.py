"""
Merges diarization speaker turns with ASR output into the shared
List[Dict] transcript schema:
  {
    "speaker_id":    str,
    "start":         float,
    "end":           float,
    "original_text": str       ← native language, filled here
    "translated_text": str     ← filled later by translation stage
  }
"""
import json
import numpy as np
import soundfile as sf
import torch

# =============================================================================
# AUDIO HELPERS
# =============================================================================

def load_audio(path: str, target_sr: int = 16000, device: str = "cpu") -> torch.Tensor:
    """Load audio file to a (1, samples) tensor on the specified device."""
    wav, sr = sf.read(path, dtype="float32")
    if wav.ndim > 1:
        wav = wav.mean(axis=1)  # stereo → mono
    if sr != target_sr:
        import scipy.signal
        num_samples = int(len(wav) * target_sr / sr)
        wav = scipy.signal.resample(wav, num_samples)
    tensor = torch.tensor(wav).unsqueeze(0)  # (1, samples)
    return tensor.to(device)


def extract_chunk(
    wav: torch.Tensor, start: float, end: float, sr: int = 16000
) -> np.ndarray:
    """Slice a waveform tensor between start and end seconds, return numpy array."""
    chunk = wav[:, int(start * sr) : int(end * sr)]
    return chunk.squeeze(0).cpu().numpy()


# =============================================================================
# TRANSCRIPT BUILDER
# =============================================================================

def build_transcript(
    audio_path: str,
    turns: list[tuple[float, float, str]],
    asr_model,
    language: str,
    device: str,
    min_duration: float = 1.5,
) -> list[dict]:
    """
    For each diarization turn, extract the audio chunk, run ASR,
    and return the structured transcript list.
    """
    from pipeline.asr.whisper_asr import transcribe_segment

    wav = load_audio(audio_path, device=device)
    transcript = []

    for i, (start, end, speaker) in enumerate(turns):
        duration = end - start
        if duration < min_duration:
            continue

        print(f"  [{i+1}/{len(turns)}] {speaker} | {start:.2f}s → {end:.2f}s ({duration:.2f}s)")

        chunk = extract_chunk(wav, start, end)
        text = transcribe_segment(
            asr_model,
            chunk,
            sample_rate=16000,
            language=language,
            device=device,
        )

        if text:
            transcript.append({
                "speaker_id":      speaker,
                "start":           round(start, 3),
                "end":             round(end, 3),
                "original_text":   text,
                "translated_text": "",
            })

    return transcript


# =============================================================================
# SAVE / LOAD HELPERS
# =============================================================================

def save_transcript(transcript: list[dict], output_path: str) -> None:
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump({"transcript": transcript}, f, ensure_ascii=False, indent=2)
    print(f"Saved {len(transcript)} segments → {output_path}")


def load_transcript(path: str) -> list[dict]:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data.get("transcript", data)
