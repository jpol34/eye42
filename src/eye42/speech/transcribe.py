"""Batch (not live-streaming) transcription of a recorded session's audio via
local faster-whisper -- no cloud API, recordings never leave the device.

Batch, not real-time streaming, by design: real-time re-segmentation makes
Whisper's worst failure mode (poor accuracy on short isolated utterances --
exactly the shape of a bid) worse, for no benefit this project needs (a
scoring engine reconstructing hand history, not live-coaching, tolerates a
few seconds to a couple minutes of lag per hand). This matches
`engine.perception`'s own existing record-then-batch precedent. Real-time
transcription is deferred, not built here -- see ROADMAP.md.

Model choice validated empirically against two real ~9-minute recorded
gameplay sessions, CPU-only (no discrete GPU): "base" + `vad_filter=True`
matches "base" alone's accuracy (both render bid-shaped numbers as digit
pairs, e.g. "6-1", unlike "tiny" which renders them as words) while running
at "tiny"'s speed (~10x real-time). See RESEARCH.md's "Speech: real-audio
Whisper validation" section for the transcript evidence.
"""

from __future__ import annotations

import math
from typing import List

from .bid_parser import Utterance

WHISPER_MODEL_SIZE = "base"
WHISPER_COMPUTE_TYPE = "int8"  # CPU-only inference; no discrete GPU on this hardware
WHISPER_VAD_FILTER = True  # cuts silence/noise before decoding -- fewer hallucinated
# segments and less wasted compute, confirmed against real audio (no accuracy cost
# observed versus running without it)


def _confidence_from_avg_logprob(avg_logprob: float) -> float:
    """faster-whisper reports each segment's ``avg_logprob`` as a natural-log
    probability (typically in roughly [-1, 0], but not hard-bounded below),
    not an already-0-1 confidence -- ``exp`` converts it back to a
    probability-like value, clamped since a very negative outlier would
    otherwise underflow to (harmlessly) exactly 0.0 rather than go negative."""
    return max(0.0, min(1.0, math.exp(avg_logprob)))


_model_cache: dict = {}  # (model_size, compute_type) -> WhisperModel; loading one is seconds-costly


def _get_model(model_size: str, compute_type: str):
    from faster_whisper import WhisperModel  # imported lazily: an optional (`speech`) extra

    key = (model_size, compute_type)
    if key not in _model_cache:
        _model_cache[key] = WhisperModel(model_size, compute_type=compute_type)
    return _model_cache[key]


def transcribe_wav(
    path: str,
    *,
    model_size: str = WHISPER_MODEL_SIZE,
    compute_type: str = WHISPER_COMPUTE_TYPE,
    vad_filter: bool = WHISPER_VAD_FILTER,
) -> List[Utterance]:
    """Transcribes a WAV file (any sample rate/channel count faster-whisper's
    underlying ffmpeg/resampler accepts -- e.g. the 44100Hz mono int16 WAV
    ``tools/live_view.py``'s ``AudioRecorder`` produces) into timestamped
    ``Utterance``s, one per Whisper segment, in chronological order."""
    model = _get_model(model_size, compute_type)
    segments, _ = model.transcribe(path, vad_filter=vad_filter)
    return [
        Utterance(
            text=segment.text.strip(),
            confidence=_confidence_from_avg_logprob(segment.avg_logprob),
            start_time=segment.start,
            end_time=segment.end,
        )
        for segment in segments
    ]
