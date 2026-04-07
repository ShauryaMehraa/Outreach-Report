"""
Handles discovery, loading, normalization, and combining of raw audio files
into a single 16kHz mono WAV ready for ASR + diarization.
"""

import glob
import os

import numpy as np
import soundfile as sf
import scipy.signal
import torch
import torchaudio


# =============================================================================
# FIND AND SORT FILES
# =============================================================================

def get_sorted_files(directory: str) -> list[str]:
    extensions = ["*.wav", "*.mp3", "*.m4a", "*.flac"]
    files = []
    for ext in extensions:
        files.extend(glob.glob(os.path.join(directory, ext)))

    if not files:
        raise FileNotFoundError(f"No audio files found in {directory}")

    narration_files = [f for f in files if "narration" in os.path.basename(f).lower()]
    other_files     = sorted(
        [f for f in files if "narration" not in os.path.basename(f).lower()],
        key=lambda f: os.path.basename(f).lower(),
    )

    ordered = narration_files + other_files

    print("File order:")
    for i, f in enumerate(ordered):
        print(f"  {i+1}. {os.path.basename(f)}")

    return ordered


# =============================================================================
# LOAD + NORMALIZE
# =============================================================================

def load_and_normalize(path: str, target_sr: int = 16000) -> torch.Tensor:
    """Load an audio file using pydub (supports mp3/m4a/flac), convert to mono, resample."""
    from pydub import AudioSegment
    import numpy as np

    audio = AudioSegment.from_file(path)
    audio = audio.set_channels(1).set_frame_rate(target_sr).set_sample_width(2)

    samples = np.array(audio.get_array_of_samples(), dtype=np.float32)
    samples /= 32768.0  # normalize to [-1, 1]

    wav = torch.tensor(samples).unsqueeze(0)  # shape: (1, samples)
    return wav


# =============================================================================
# COMBINE + SAVE
# =============================================================================

def combine_audio(
    directory: str,
    output_path: str,
    target_sr: int = 16000,
) -> str:
    files = get_sorted_files(directory)

    chunks = []
    for path in files:
        print(f"Loading: {os.path.basename(path)}")
        wav = load_and_normalize(path, target_sr)
        duration = wav.shape[1] / target_sr
        print(f"  → {duration:.2f}s | shape: {wav.shape}")
        chunks.append(wav)

    combined = torch.cat(chunks, dim=1)
    total_duration = combined.shape[1] / target_sr
    print(f"\nCombined duration: {total_duration:.2f}s")

    import soundfile as sf
    sf.write(
        output_path,
        combined.squeeze(0).numpy(),
        target_sr,
        subtype="PCM_16",
    )
    if False:  # disabled torchaudio.save
        import soundfile as sf
    sf.write(output_path, combined.squeeze(0).numpy(), target_sr, subtype="PCM_16")
    print(f"Saved → {output_path}")
    return output_path
