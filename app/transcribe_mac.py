from __future__ import annotations

import logging
import math
import os
import platform
import tempfile
import re
from pathlib import Path
from dataclasses import dataclass
from typing import Any, List, Literal, Optional

from faster_whisper import WhisperModel

logger = logging.getLogger(__name__)

RequestedDevice = Literal["auto", "apple_gpu", "cpu"]
ResolvedDevice = Literal["apple_gpu", "cpu"]


@dataclass
class EngineDebugInfo:
    requested: RequestedDevice
    resolved: ResolvedDevice
    backend: str
    reason: str


@dataclass
class SegmentResult:
    id: int
    start: float
    end: float
    text: str


@dataclass
class TranscriptionResult:
    text: str
    language: Optional[str]
    duration: Optional[float]
    segments: List[SegmentResult]
    debug: Optional[dict[str, str]] = None


class GpuNotAvailableError(RuntimeError):
    def __init__(self, reason: str) -> None:
        self.reason = reason
        super().__init__(f"GPU not available: {reason}")


_MODEL_CACHE: dict[str, WhisperModel] = {}
_MLX_MODEL_NAME_ALIASES: dict[str, str] = {
    "tiny": "mlx-community/whisper-tiny-mlx",
    "tiny.en": "mlx-community/whisper-tiny.en-mlx",
    "base": "mlx-community/whisper-base-mlx",
    "base.en": "mlx-community/whisper-base.en-mlx",
    "small": "mlx-community/whisper-small-mlx",
    "small.en": "mlx-community/whisper-small.en-mlx",
    "medium": "mlx-community/whisper-medium-mlx",
    "medium.en": "mlx-community/whisper-medium.en-mlx",
    "large-v2": "mlx-community/whisper-large-v2-mlx",
    "large-v3": "mlx-community/whisper-large-v3-mlx",
    "large-v3-turbo": "mlx-community/whisper-large-v3-turbo",
}
_MLX_REPO_ALIASES: dict[str, str] = {
    # SenseVoiceSmall is not a CTranslate2 Whisper checkpoint and may not be
    # directly consumable by mlx-whisper in all environments.
    "funaudiollm/sensevoicesmall": "mlx-community/whisper-small-mlx",
}
_CPU_MODEL_NAME_ALIASES: dict[str, str] = {
    "funaudiollm/sensevoicesmall": "small",
}
_SENSEVOICE_REPO_ID = "funaudiollm/sensevoicesmall"
_SENSEVOICE_MODEL_CACHE: dict[str, Any] = {}


def _canonicalize_model_name(model_name: str) -> str:
    normalized = model_name.strip().rstrip("/")
    if normalized.lower().startswith("https://huggingface.co/"):
        normalized = normalized[len("https://huggingface.co/") :].strip("/")
    if "/revision/" in normalized.lower():
        normalized = re.split(r"/revision/", normalized, maxsplit=1, flags=re.IGNORECASE)[0]
    return normalized


def _normalize_cpu_model_name(model_name: str) -> str:
    normalized = _canonicalize_model_name(model_name)
    alias = _CPU_MODEL_NAME_ALIASES.get(normalized.lower())
    if alias:
        logger.warning("whisper_cpu_model_alias requested=%s resolved=%s", model_name, alias)
        return alias
    return normalized


def _is_sensevoice_model(model_name: str) -> bool:
    return _canonicalize_model_name(model_name).lower() == _SENSEVOICE_REPO_ID


def _normalize_sensevoice_language(language: Optional[str]) -> str:
    if not language:
        return "auto"
    normalized = language.strip().lower()
    mapping = {
        "zh": "zn",
        "zh-cn": "zn",
        "zh-tw": "zn",
        "cmn": "zn",
    }
    return mapping.get(normalized, normalized)


def _sensevoice_verbatim_text(raw_text: str) -> str:
    # Keep transcript close to original words: remove control tokens only.
    cleaned = re.sub(r"<\|[^|]+\|>", " ", raw_text)
    return re.sub(r"\s+", " ", cleaned).strip()


def _sensevoice_join_tokens(tokens: List[str]) -> str:
    joined = ""
    no_space_before = {".", ",", "!", "?", ":", ";", ")", "]", "}", "。", "，", "！", "？", "：", "；", "、"}
    no_space_after = {"(", "[", "{"}
    for token in tokens:
        t = token.strip()
        if not t:
            continue
        if not joined:
            joined = t
            continue
        if t in no_space_before or joined[-1] in no_space_after:
            joined += t
            continue
        joined += f" {t}"
    return re.sub(r"\s+", " ", joined).strip()


def _sensevoice_parse_timestamps(item: dict[str, Any]) -> List[tuple[str, float, float]]:
    parsed: List[tuple[str, float, float]] = []
    raw = item.get("timestamp")
    if not isinstance(raw, list):
        return parsed
    for entry in raw:
        if not isinstance(entry, (list, tuple)) or len(entry) < 3:
            continue
        token = str(entry[0]).strip()
        if not token:
            continue
        try:
            start = float(entry[1])
            end = float(entry[2])
        except (TypeError, ValueError):
            continue
        if end < start:
            end = start
        parsed.append((token, start, end))
    return parsed


def _sensevoice_parse_sentence_segments(item: dict[str, Any]) -> List[SegmentResult]:
    raw = item.get("sentence_info")
    if not isinstance(raw, list):
        return []

    segments: List[SegmentResult] = []
    for block in raw:
        if not isinstance(block, dict):
            continue

        text = _sensevoice_verbatim_text(str(block.get("text", "")))
        if not text:
            continue

        start = 0.0
        end = 0.0
        ts = block.get("timestamp")
        if isinstance(ts, list) and ts:
            first = ts[0]
            last = ts[-1]
            if isinstance(first, (list, tuple)) and len(first) >= 3:
                try:
                    start = float(first[1])
                except (TypeError, ValueError):
                    start = 0.0
            if isinstance(last, (list, tuple)) and len(last) >= 3:
                try:
                    end = float(last[2])
                except (TypeError, ValueError):
                    end = start
        if end < start:
            end = start

        segments.append(SegmentResult(id=len(segments), start=start, end=end, text=text))

    return segments


def _sensevoice_build_segments(timed_tokens: List[tuple[str, float, float]]) -> List[SegmentResult]:
    if not timed_tokens:
        return []

    sentence_endings = {".", "!", "?", "。", "！", "？"}
    segments: List[SegmentResult] = []
    current_tokens: List[str] = []
    current_start = timed_tokens[0][1]
    current_end = timed_tokens[0][2]

    def flush_segment() -> None:
        nonlocal current_tokens, current_start, current_end
        text = _sensevoice_join_tokens(current_tokens)
        if text:
            segments.append(
                SegmentResult(
                    id=len(segments),
                    start=current_start,
                    end=current_end,
                    text=text,
                )
            )
        current_tokens = []

    for token, start, end in timed_tokens:
        if not current_tokens:
            current_start = start
            current_end = end
            current_tokens.append(token)
            continue

        current_tokens.append(token)
        current_end = end

        segment_duration = current_end - current_start
        current_text = _sensevoice_join_tokens(current_tokens)
        should_flush = False
        if token in sentence_endings and segment_duration >= 1.0:
            should_flush = True
        elif segment_duration >= 3.0:
            should_flush = True
        elif len(current_text) >= 28 and token in {",", ";", "，", "；", "、"}:
            should_flush = True

        if should_flush:
            flush_segment()

    flush_segment()
    return segments


def _get_sensevoice_model(model_name: str, device: str) -> Any:
    from funasr import AutoModel

    canonical = _canonicalize_model_name(model_name)
    cache_key = f"{canonical}|{device}"
    if cache_key not in _SENSEVOICE_MODEL_CACHE:
        logger.info("sensevoice_model_init cache_miss model=%s device=%s", canonical, device)
        _SENSEVOICE_MODEL_CACHE[cache_key] = AutoModel(
            model=canonical,
            device=device,
            hub="hf",
            trust_remote_code=True,
        )
    else:
        logger.info("sensevoice_model_init cache_hit model=%s device=%s", canonical, device)
    return _SENSEVOICE_MODEL_CACHE[cache_key]


def _transcribe_with_sensevoice(
    temp_path: str,
    model_name: str,
    language: Optional[str],
    require_gpu: bool,
) -> tuple[str, Optional[str], Optional[float], List[SegmentResult], EngineDebugInfo]:
    device_candidates = ["mps", "cpu"]
    if require_gpu:
        device_candidates = ["mps"]

    last_exc: Optional[Exception] = None
    for device in device_candidates:
        try:
            model = _get_sensevoice_model(model_name, device)
            result = model.generate(
                input=temp_path,
                cache={},
                language=_normalize_sensevoice_language(language),
                use_itn=True,
                output_timestamp=True,
                batch_size_s=60,
                merge_vad=False,
                merge_length_s=15,
            )

            text_parts: List[str] = []
            sentence_segments: List[SegmentResult] = []
            timed_tokens: List[tuple[str, float, float]] = []
            if isinstance(result, list):
                for item in result:
                    if isinstance(item, dict):
                        text_raw = str(item.get("text", ""))
                        text_clean = _sensevoice_verbatim_text(text_raw)
                        if text_clean:
                            text_parts.append(text_clean)
                        sentence_segments.extend(_sensevoice_parse_sentence_segments(item))
                        timed_tokens.extend(_sensevoice_parse_timestamps(item))
            elif isinstance(result, dict):
                text_raw = str(result.get("text", ""))
                text_clean = _sensevoice_verbatim_text(text_raw)
                if text_clean:
                    text_parts.append(text_clean)
                sentence_segments.extend(_sensevoice_parse_sentence_segments(result))
                timed_tokens.extend(_sensevoice_parse_timestamps(result))

            segments = sentence_segments or _sensevoice_build_segments(timed_tokens)
            if segments:
                text = " ".join(seg.text for seg in segments).strip()
                duration = max(seg.end for seg in segments)
            else:
                text = " ".join(text_parts).strip()
                duration = None
            if not text:
                raise RuntimeError("SenseVoice returned empty transcript")
            if not segments:
                raise RuntimeError("SenseVoice did not return timestamp segments")

            resolved = "apple_gpu" if device == "mps" else "cpu"
            reason = "sensevoice_mps" if device == "mps" else "sensevoice_cpu"
            debug = EngineDebugInfo(
                requested="apple_gpu" if device == "mps" else "cpu",
                resolved=resolved,
                backend="funasr",
                reason=reason,
            )
            return text, language, duration, segments, debug
        except Exception as exc:
            last_exc = exc
            logger.warning(
                "sensevoice_transcribe_failed model=%s device=%s err_type=%s err=%s",
                model_name,
                device,
                exc.__class__.__name__,
                exc,
            )
            continue

    raise RuntimeError(f"SenseVoice inference failed: {last_exc}")


def _resolve_device() -> RequestedDevice:
    raw = os.getenv("WHISPER_DEVICE", "auto").strip().lower()
    mapping = {
        "intel_gpu": "apple_gpu",
        "apple_gpu": "apple_gpu",
        "mac_gpu": "apple_gpu",
    }
    raw = mapping.get(raw, raw)
    if raw in {"auto", "apple_gpu", "cpu"}:
        return raw
    logger.warning("Unknown WHISPER_DEVICE=%s, fallback to auto", raw)
    return "auto"


def _cpu_threads_target() -> int:
    cpu_count = os.cpu_count() or 1
    raw_ratio = os.getenv("WHISPER_CPU_USAGE_RATIO", "0.8").strip()
    try:
        ratio = float(raw_ratio)
    except ValueError:
        logger.warning("Invalid WHISPER_CPU_USAGE_RATIO=%s, fallback to 0.8", raw_ratio)
        ratio = 0.8
    ratio = min(max(ratio, 0.1), 1.0)
    threads = max(1, math.floor(cpu_count * ratio))
    logger.info(
        "whisper_cpu_threads_config cpu_count=%s ratio=%.2f threads=%s",
        cpu_count,
        ratio,
        threads,
    )
    return threads


def _is_macos_arm64() -> bool:
    return platform.system() == "Darwin" and platform.machine() in {"arm64", "aarch64"}


def _probe_mlx_gpu() -> tuple[bool, str]:
    if not _is_macos_arm64():
        return False, "platform_not_macos_arm64"

    try:
        import mlx.core as mx
    except Exception as exc:
        return False, f"mlx_unavailable:{exc.__class__.__name__}"

    try:
        default_device = str(mx.default_device())
        if "gpu" in default_device.lower():
            return True, f"mlx_gpu_available:{default_device}"
        return False, f"mlx_gpu_not_selected:{default_device}"
    except Exception as exc:
        return False, f"mlx_probe_failed:{exc.__class__.__name__}:{exc}"


def check_gpu_support() -> tuple[bool, str]:
    return _probe_mlx_gpu()


def _resolve_engine(requested: RequestedDevice) -> EngineDebugInfo:
    if requested == "cpu":
        return EngineDebugInfo(requested=requested, resolved="cpu", backend="ctranslate2", reason="forced_cpu")

    available, reason = _probe_mlx_gpu()
    if available:
        return EngineDebugInfo(requested=requested, resolved="apple_gpu", backend="mlx-whisper", reason=reason)

    return EngineDebugInfo(requested=requested, resolved="cpu", backend="ctranslate2", reason=reason)


def _model_cache_key(model_name: str, cpu_threads: int) -> str:
    return f"{model_name}|cpu|int8|cpu_threads={cpu_threads}"


def _get_cpu_model(model_name: str) -> WhisperModel:
    resolved_model_name = _normalize_cpu_model_name(model_name)
    cpu_threads = _cpu_threads_target()
    cache_key = _model_cache_key(resolved_model_name, cpu_threads)
    if cache_key not in _MODEL_CACHE:
        logger.info(
            "whisper_cpu_model_init cache_miss requested=%s resolved=%s device=cpu compute_type=int8 cpu_threads=%s",
            model_name,
            resolved_model_name,
            cpu_threads,
        )
        _MODEL_CACHE[cache_key] = WhisperModel(
            resolved_model_name,
            device="cpu",
            compute_type="int8",
            cpu_threads=cpu_threads,
        )
    else:
        logger.info(
            "whisper_cpu_model_init cache_hit requested=%s resolved=%s device=cpu compute_type=int8 cpu_threads=%s",
            model_name,
            resolved_model_name,
            cpu_threads,
        )
    return _MODEL_CACHE[cache_key]


def _mlx_model_candidates(model_name: str) -> List[str]:
    normalized = _canonicalize_model_name(model_name)

    repo_alias = _MLX_REPO_ALIASES.get(normalized.lower())
    if repo_alias and repo_alias != normalized:
        return [normalized, repo_alias]

    if "/" in normalized:
        return [normalized]

    alias = _MLX_MODEL_NAME_ALIASES.get(normalized)
    if alias and alias != normalized:
        return [alias, normalized]
    return [normalized]


def _is_hf_repo_access_error(exc: Exception) -> bool:
    exc_name = exc.__class__.__name__.lower()
    message = str(exc).lower()
    hf_error_keywords = {
        "repositorynotfounderror",
        "gatedrepoerror",
        "revisionnotfounderror",
        "entrynotfounderror",
        "hfhubhttperror",
    }
    return (
        exc_name in hf_error_keywords
        or "401" in message
        or "repository not found" in message
        or "unauthorized" in message
    )


def _transcribe_with_mlx(
    temp_path: str,
    model_name: str,
    language: Optional[str],
    prompt: Optional[str],
    temperature: float,
) -> tuple[str, Optional[str], Optional[float], List[SegmentResult]]:
    import mlx_whisper

    last_exc: Optional[Exception] = None
    result: Any = None
    for idx, candidate in enumerate(_mlx_model_candidates(model_name)):
        try:
            logger.info("whisper_apple_gpu_model_try requested=%s candidate=%s", model_name, candidate)
            result = mlx_whisper.transcribe(
                temp_path,
                path_or_hf_repo=candidate,
                language=language,
                initial_prompt=prompt,
                temperature=temperature,
            )
            break
        except Exception as exc:
            last_exc = exc
            if idx == 0 and candidate != model_name and _is_hf_repo_access_error(exc):
                logger.warning(
                    "whisper_apple_gpu_model_alias_retry requested=%s candidate=%s err_type=%s err=%s",
                    model_name,
                    candidate,
                    exc.__class__.__name__,
                    exc,
                )
                continue
            raise

    if result is None and last_exc is not None:
        raise last_exc

    text = str(result.get("text", "")).strip()
    language_out = result.get("language")

    segments: List[SegmentResult] = []
    for idx, seg in enumerate(result.get("segments", []) or []):
        cleaned = str(seg.get("text", "")).strip()
        segments.append(
            SegmentResult(
                id=idx,
                start=float(seg.get("start", 0.0)),
                end=float(seg.get("end", 0.0)),
                text=cleaned,
            )
        )

    duration = None
    if segments:
        duration = max(s.end for s in segments)

    return text, language_out, duration, segments


def _transcribe_with_cpu_whisper(
    temp_path: str,
    model_name: str,
    language: Optional[str],
    prompt: Optional[str],
    temperature: float,
) -> tuple[str, Optional[str], Optional[float], List[SegmentResult]]:
    model = _get_cpu_model(model_name)
    raw_segments, info = model.transcribe(
        temp_path,
        language=language,
        initial_prompt=prompt,
        temperature=temperature,
    )

    segment_results: List[SegmentResult] = []
    texts: List[str] = []
    for idx, seg in enumerate(raw_segments):
        cleaned = seg.text.strip()
        if cleaned:
            texts.append(cleaned)
        segment_results.append(SegmentResult(id=idx, start=float(seg.start), end=float(seg.end), text=cleaned))

    return (
        " ".join(texts).strip(),
        getattr(info, "language", language),
        getattr(info, "duration", None),
        segment_results,
    )


def transcribe_audio(
    audio_bytes: bytes,
    model_name: str,
    language: Optional[str] = None,
    prompt: Optional[str] = None,
    temperature: float = 0.0,
    include_debug: bool = False,
    require_gpu: bool = False,
    source_filename: Optional[str] = None,
) -> TranscriptionResult:
    requested = _resolve_device()
    engine = _resolve_engine(requested)
    logger.info(
        "whisper_engine_resolved requested=%s resolved=%s backend=%s reason=%s require_gpu=%s",
        engine.requested,
        engine.resolved,
        engine.backend,
        engine.reason,
        require_gpu,
    )
    if require_gpu and engine.resolved != "apple_gpu":
        raise GpuNotAvailableError(engine.reason)

    temp_suffix = Path(source_filename or "").suffix or ".tmp"
    with tempfile.NamedTemporaryFile(delete=False, suffix=temp_suffix) as tmp_file:
        tmp_file.write(audio_bytes)
        temp_path = tmp_file.name

    try:
        if _is_sensevoice_model(model_name):
            text, language_out, duration, segments, engine = _transcribe_with_sensevoice(
                temp_path=temp_path,
                model_name=model_name,
                language=language,
                require_gpu=require_gpu,
            )
        elif engine.resolved == "apple_gpu":
            try:
                logger.info("whisper_apple_gpu_transcribe_start model=%s", model_name)
                text, language_out, duration, segments = _transcribe_with_mlx(
                    temp_path=temp_path,
                    model_name=model_name,
                    language=language,
                    prompt=prompt,
                    temperature=temperature,
                )
                logger.info("whisper_apple_gpu_transcribe_ok model=%s", model_name)
            except Exception as exc:
                logger.warning(
                    "whisper_apple_gpu_transcribe_failed fallback_to_cpu err_type=%s err=%s",
                    exc.__class__.__name__,
                    exc,
                )
                engine = EngineDebugInfo(
                    requested=engine.requested,
                    resolved="cpu",
                    backend="ctranslate2",
                    reason=f"apple_gpu_fallback_to_cpu:{exc.__class__.__name__}",
                )
                text, language_out, duration, segments = _transcribe_with_cpu_whisper(
                    temp_path=temp_path,
                    model_name=model_name,
                    language=language,
                    prompt=prompt,
                    temperature=temperature,
                )
        else:
            text, language_out, duration, segments = _transcribe_with_cpu_whisper(
                temp_path=temp_path,
                model_name=model_name,
                language=language,
                prompt=prompt,
                temperature=temperature,
            )

        logger.info("whisper_engine_final requested=%s resolved=%s backend=%s reason=%s", engine.requested, engine.resolved, engine.backend, engine.reason)
        return TranscriptionResult(
            text=text,
            language=language_out,
            duration=duration,
            segments=segments,
            debug=engine.__dict__ if include_debug else None,
        )
    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)
