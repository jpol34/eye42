from __future__ import annotations

import math
import wave
from pathlib import Path

import pytest

from eye42.speech.transcribe import _confidence_from_avg_logprob, _get_model, transcribe_wav


def _tiny_model_or_skip():
    """faster-whisper downloads model weights from Hugging Face Hub on first
    use -- skip rather than fail these two real-model tests in an
    environment with no cached weights and no network access (a
    network-restricted CI runner, a fresh offline sandbox), instead of
    blocking the rest of the suite on an external resource being reachable."""
    try:
        return _get_model("tiny", "int8")
    except Exception as exc:  # pragma: no cover -- exercised only when offline
        pytest.skip(f"faster-whisper 'tiny' model unavailable (no network/cache?): {exc}")


def test_confidence_from_avg_logprob_converts_log_probability():
    assert _confidence_from_avg_logprob(0.0) == 1.0  # log(1.0) == 0.0 -- perfect confidence
    assert math.isclose(_confidence_from_avg_logprob(-0.5), math.exp(-0.5))


def test_confidence_from_avg_logprob_is_clamped_to_zero_one():
    assert 0.0 <= _confidence_from_avg_logprob(-100.0) < 1e-30  # a very negative logprob is near-zero confidence
    assert _confidence_from_avg_logprob(5.0) == 1.0  # a log-probability should never be positive, but clamp anyway


def test_get_model_reuses_the_same_instance_for_the_same_config():
    _tiny_model_or_skip()
    first = _get_model("tiny", "int8")
    second = _get_model("tiny", "int8")
    assert first is second


def _write_silence_wav(path: Path, seconds: float = 1.0, samplerate: int = 16000) -> None:
    with wave.open(str(path), "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(samplerate)
        wav.writeframes(b"\x00\x00" * int(seconds * samplerate))


def test_transcribe_wav_runs_end_to_end_on_silence(tmp_path):
    """Not an accuracy test (real accuracy is validated against real
    recorded gameplay audio, documented in RESEARCH.md, not committed here
    as a 45MB+ test fixture) -- this only exercises the real faster-whisper
    plumbing (model load, VAD filter, segment -> Utterance conversion) end
    to end without needing real speech, using the smallest model."""
    _tiny_model_or_skip()
    wav_path = tmp_path / "silence.wav"
    _write_silence_wav(wav_path)

    utterances = transcribe_wav(str(wav_path), model_size="tiny")

    assert isinstance(utterances, list)
    for utterance in utterances:
        assert 0.0 <= utterance.confidence <= 1.0
        assert utterance.end_time >= utterance.start_time
