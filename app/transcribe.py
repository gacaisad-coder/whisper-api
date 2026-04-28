from __future__ import annotations

import logging
import math
import os
import platform
import re
import tempfile
import json
from pathlib import Path
from dataclasses import asdict, dataclass
from typing import Any, List, Literal, Optional

from faster_whisper import WhisperModel

logger = logging.getLogger(__name__)


RequestedDevice = Literal["auto", "intel_gpu", "cpu"]
ResolvedDevice = Literal["intel_gpu", "cpu"]


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
_MODEL_NAME_ALIASES: dict[str, str] = {
    "openvino/whisper-large-v3-int8-ov": "large-v3",
    "openvino/whisper-large-v3-fp16-ov": "large-v3",
    "openvino/whisper-small-int8-ov": "small",
    "openvino/whisper-base-int8-ov": "base",
    "openvino/whisper-tiny-int8-ov": "tiny",
    "funaudiollm/sensevoicesmall": "small",
}


def _resolve_device() -> RequestedDevice:
    raw = os.getenv("WHISPER_DEVICE", "auto").strip().lower()
    if raw in {"auto", "intel_gpu", "cpu"}:
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


def _probe_openvino_gpu() -> tuple[bool, str]:
    try:
        from openvino import Core
    except Exception as exc:  # pragma: no cover - runtime dependent
        try:
            from openvino.runtime import Core
        except Exception:
            return False, f"openvino.runtime_unavailable:{exc.__class__.__name__}"

    try:
        devices = list(Core().available_devices)
    except Exception as exc:  # pragma: no cover - runtime dependent
        return False, f"openvino.probe_failed:{exc.__class__.__name__}:{exc}"

    if "GPU" in devices:
        return True, f"openvino_gpu_available:{devices}"
    return False, f"openvino_gpu_missing:{devices}"


def check_gpu_support() -> tuple[bool, str]:
    return _probe_openvino_gpu()


def _resolve_engine(requested: RequestedDevice) -> EngineDebugInfo:
    if requested == "cpu":
        return EngineDebugInfo(
            requested=requested,
            resolved="cpu",
            backend="ctranslate2",
            reason="forced_cpu",
        )

    available, reason = _probe_openvino_gpu()
    if available:
        return EngineDebugInfo(
            requested=requested,
            resolved="intel_gpu",
            backend="openvino",
            reason=reason,
        )

    return EngineDebugInfo(
        requested=requested,
        resolved="cpu",
        backend="ctranslate2",
        reason=reason,
    )


def _model_runtime_args(engine: EngineDebugInfo, compute_type: Optional[str] = None) -> tuple[str, str]:
    if engine.resolved == "intel_gpu":
        return "auto", compute_type or "int8_float16"
    return "cpu", "int8"


def _model_cache_key(model_name: str, device: str, compute_type: str, cpu_threads: int) -> str:
    return f"{model_name}|{device}|{compute_type}|cpu_threads={cpu_threads}"


def _get_model(model_name: str, engine: EngineDebugInfo, compute_type: Optional[str] = None) -> WhisperModel:
    device, compute_type = _model_runtime_args(engine, compute_type=compute_type)
    cpu_threads = _cpu_threads_target()
    cache_key = _model_cache_key(model_name, device, compute_type, cpu_threads)
    if cache_key not in _MODEL_CACHE:
        logger.info(
            "whisper_model_init cache_miss model=%s requested=%s resolved=%s backend=%s device=%s compute_type=%s cpu_threads=%s",
            model_name,
            engine.requested,
            engine.resolved,
            engine.backend,
            device,
            compute_type,
            cpu_threads,
        )
        _MODEL_CACHE[cache_key] = WhisperModel(
            model_name,
            device=device,
            compute_type=compute_type,
            cpu_threads=cpu_threads,
        )
    else:
        logger.info(
            "whisper_model_init cache_hit model=%s requested=%s resolved=%s backend=%s device=%s compute_type=%s cpu_threads=%s",
            model_name,
            engine.requested,
            engine.resolved,
            engine.backend,
            device,
            compute_type,
            cpu_threads,
        )
    return _MODEL_CACHE[cache_key]


def _canonicalize_model_name(model_name: str) -> str:
    normalized = model_name.strip().rstrip("/")
    if normalized.lower().startswith("https://huggingface.co/"):
        normalized = normalized[len("https://huggingface.co/") :].strip("/")

    if "/revision/" in normalized.lower():
        normalized = re.split(r"/revision/", normalized, maxsplit=1, flags=re.IGNORECASE)[0]
    return normalized


def _parse_hf_repo_and_revision(model_name: str) -> tuple[str, Optional[str]]:
    normalized = model_name.strip().rstrip("/")
    if normalized.lower().startswith("https://huggingface.co/"):
        normalized = normalized[len("https://huggingface.co/") :].strip("/")

    match = re.search(r"/revision/([^/]+)$", normalized, flags=re.IGNORECASE)
    if match:
        repo_id = normalized[: match.start()]
        revision = match.group(1)
        return repo_id, revision
    return normalized, None


def _normalize_model_name(model_name: str) -> str:
    normalized = _canonicalize_model_name(model_name)
    lowered = normalized.lower()
    alias = _MODEL_NAME_ALIASES.get(lowered)
    if alias:
        logger.info("whisper_model_alias requested=%s resolved=%s", model_name, alias)
        return alias

    openvino_prefix = "openvino/whisper-"
    if lowered.startswith(openvino_prefix) and lowered.endswith("-ov"):
        model_hint = lowered[len(openvino_prefix) : -len("-ov")]
        model_hint = re.sub(r"-(int8|fp16|fp32)$", "", model_hint)
        for candidate in (
            "large-v3-turbo",
            "large-v3",
            "large-v2",
            "large-v1",
            "medium",
            "small",
            "base",
            "tiny",
        ):
            if model_hint.startswith(candidate):
                logger.info("whisper_model_alias requested=%s inferred=%s", model_name, candidate)
                return candidate

    return normalized


def _is_unsupported_gpu_compute_error(exc: Exception) -> bool:
    message = str(exc).lower()
    return "int8_float16" in message and "do not support efficient" in message


def _gpu_compute_type_candidates() -> List[str]:
    # On Windows Intel GPU with OpenVINO, int8 is often more broadly supported
    # than int8_float16. Keep int8_float16 first on non-Windows for performance.
    if platform.system() == "Windows":
        return ["int8", "int8_float16"]
    return ["int8_float16", "int8"]


def _fallback_to_cpu(engine: EngineDebugInfo, reason: str) -> EngineDebugInfo:
    return EngineDebugInfo(
        requested=engine.requested,
        resolved="cpu",
        backend="ctranslate2",
        reason=f"gpu_fallback_to_cpu:{reason}",
    )


def _fallback_to_faster_whisper_gpu(engine: EngineDebugInfo, reason: str) -> EngineDebugInfo:
    return EngineDebugInfo(
        requested=engine.requested,
        resolved="intel_gpu",
        backend="ctranslate2",
        reason=f"openvino_model_fallback_to_faster_whisper:{reason}",
    )


def _is_openvino_repo_model(model_name: str) -> bool:
    repo_id, _ = _parse_hf_repo_and_revision(model_name)
    return repo_id.lower().startswith("openvino/")


def _download_openvino_model_snapshot(model_name: str) -> str:
    from huggingface_hub import snapshot_download

    repo_id, revision = _parse_hf_repo_and_revision(model_name)
    cache_root = Path(os.getenv("WHISPER_OPENVINO_CACHE_DIR", Path.home() / ".cache" / "whisper-api"))
    model_cache_dir = cache_root / repo_id.replace("/", "--")
    model_cache_dir.mkdir(parents=True, exist_ok=True)

    token = os.getenv("HF_TOKEN") or os.getenv("HUGGINGFACE_HUB_TOKEN")
    download_kwargs: dict[str, Any] = {
        "repo_id": repo_id,
        "local_dir": str(model_cache_dir),
        "local_dir_use_symlinks": False,
    }
    if revision:
        download_kwargs["revision"] = revision
    if token:
        download_kwargs["token"] = token

    logger.info(
        "whisper_openvino_snapshot_download repo_id=%s revision=%s local_dir=%s token_set=%s",
        repo_id,
        revision or "main",
        model_cache_dir,
        bool(token),
    )
    snapshot_download(**download_kwargs)
    return str(model_cache_dir)


def _coerce_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _extract_chunk_times(chunk: Any) -> tuple[float, float]:
    # Compatibility for different openvino-genai chunk fields across versions.
    direct_start = getattr(chunk, "start", None)
    direct_end = getattr(chunk, "end", None)
    if direct_start is not None or direct_end is not None:
        return _coerce_float(direct_start, 0.0), _coerce_float(direct_end, _coerce_float(direct_start, 0.0))

    for start_key, end_key in (("start_ts", "end_ts"), ("start_time", "end_time")):
        if hasattr(chunk, start_key) or hasattr(chunk, end_key):
            start = _coerce_float(getattr(chunk, start_key, 0.0), 0.0)
            end = _coerce_float(getattr(chunk, end_key, start), start)
            return start, end

    timestamp = getattr(chunk, "timestamp", None)
    if isinstance(timestamp, (list, tuple)) and len(timestamp) >= 2:
        start = _coerce_float(timestamp[0], 0.0)
        end = _coerce_float(timestamp[1], start)
        return start, end
    if isinstance(timestamp, dict):
        start = _coerce_float(timestamp.get("start", 0.0), 0.0)
        end = _coerce_float(timestamp.get("end", start), start)
        return start, end

    if isinstance(chunk, dict):
        start = _coerce_float(chunk.get("start", chunk.get("start_ts", 0.0)), 0.0)
        end = _coerce_float(chunk.get("end", chunk.get("end_ts", start)), start)
        return start, end

    return 0.0, 0.0


def _transcribe_with_openvino_model(
    temp_path: str,
    model_name: str,
    language: Optional[str],
    prompt: Optional[str],
    temperature: float,
) -> tuple[str, Optional[str], Optional[float], List[SegmentResult]]:
    import openvino_genai as ov_genai
    from faster_whisper.audio import decode_audio

    model_path = _download_openvino_model_snapshot(model_name)
    pipe = ov_genai.WhisperPipeline(model_path, "GPU")
    audio_input = decode_audio(temp_path)
    generate_kwargs: dict[str, Any] = {"temperature": temperature, "return_timestamps": True}
    # openvino-genai rejects None for some optional args; use safe defaults.
    if language:
        supported_languages = _load_openvino_supported_languages(model_path)
        resolved_language = _resolve_openvino_language(language, supported_languages)
        if resolved_language:
            generate_kwargs["language"] = resolved_language
        else:
            logger.warning(
                "whisper_openvino_language_not_supported use_auto_detect model=%s language=%s supported_count=%s",
                model_name,
                language,
                len(supported_languages),
            )
    generate_kwargs["prompt"] = prompt or ""
    try:
        result: Any = pipe.generate(audio_input, **generate_kwargs)
    except TypeError as exc:
        # Some openvino-genai versions do not support return_timestamps kwargs.
        if "return_timestamps" not in generate_kwargs:
            raise
        logger.warning(
            "whisper_openvino_no_return_timestamps_kwarg model=%s err_type=%s err=%s",
            model_name,
            exc.__class__.__name__,
            exc,
        )
        generate_kwargs.pop("return_timestamps", None)
        result = pipe.generate(audio_input, **generate_kwargs)
    except RuntimeError as exc:
        err_msg = str(exc)
        if "lang_to_id" not in err_msg or "language" not in err_msg or "language" not in generate_kwargs:
            raise
        logger.warning(
            "whisper_openvino_language_unsupported retry_without_language model=%s language=%s err=%s",
            model_name,
            generate_kwargs.get("language"),
            err_msg,
        )
        generate_kwargs.pop("language", None)
        result = pipe.generate(audio_input, **generate_kwargs)

    text = ""
    language_out: Optional[str] = language
    duration: Optional[float] = None
    segments: List[SegmentResult] = []

    if hasattr(result, "texts") and getattr(result, "texts"):
        text = str(result.texts[0]).strip()
    elif hasattr(result, "text"):
        text = str(result.text).strip()
    else:
        text = str(result).strip()

    if hasattr(result, "language"):
        language_out = getattr(result, "language")

    raw_chunks = getattr(result, "chunks", None) or getattr(result, "segments", None)
    if raw_chunks:
        for idx, chunk in enumerate(raw_chunks):
            start, end = _extract_chunk_times(chunk)
            if isinstance(chunk, dict):
                chunk_text = str(chunk.get("text", "")).strip()
            else:
                chunk_text = str(getattr(chunk, "text", "")).strip()
            segments.append(SegmentResult(id=idx, start=start, end=end, text=chunk_text))
        if segments:
            duration = max(s.end for s in segments)
    elif text:
        segments.append(SegmentResult(id=0, start=0.0, end=0.0, text=text))

    return text, language_out, duration, segments


def _load_openvino_supported_languages(model_path: str) -> set[str]:
    config_path = Path(model_path) / "generation_config.json"
    if not config_path.exists():
        return set()
    try:
        payload = json.loads(config_path.read_text(encoding="utf-8"))
    except Exception as exc:
        logger.warning("whisper_openvino_generation_config_parse_failed path=%s err=%s", config_path, exc)
        return set()
    lang_to_id = payload.get("lang_to_id")
    if isinstance(lang_to_id, dict):
        return {str(k) for k in lang_to_id.keys()}
    return set()


def _resolve_openvino_language(language: str, supported_languages: set[str]) -> Optional[str]:
    normalized_input = language.strip().lower()
    alias_map = {
        "zh-cn": "zh",
        "zh-tw": "zh",
        "en-us": "en",
        "en-gb": "en",
        "pt-br": "pt",
    }
    candidates = [normalized_input]
    if normalized_input in alias_map:
        candidates.append(alias_map[normalized_input])

    if not supported_languages:
        return candidates[0]

    normalized_supported: dict[str, str] = {}
    for supported in supported_languages:
        normalized_supported[supported.lower()] = supported
        token_match = re.fullmatch(r"<\|([a-z]{2,3})\|>", supported.lower())
        if token_match:
            normalized_supported[token_match.group(1)] = supported

    for candidate in candidates:
        if candidate in normalized_supported:
            return normalized_supported[candidate]
    return None


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
    requested_model_name = _canonicalize_model_name(model_name)
    model_name = _normalize_model_name(model_name)
    logger.info(
        "whisper_engine_resolved requested=%s resolved=%s backend=%s reason=%s require_gpu=%s requested_model=%s model=%s",
        engine.requested,
        engine.resolved,
        engine.backend,
        engine.reason,
        require_gpu,
        requested_model_name,
        model_name,
    )
    if require_gpu and engine.resolved != "intel_gpu":
        raise GpuNotAvailableError(engine.reason)

    temp_suffix = Path(source_filename or "").suffix or ".tmp"
    with tempfile.NamedTemporaryFile(delete=False, suffix=temp_suffix) as tmp_file:
        tmp_file.write(audio_bytes)
        temp_path = tmp_file.name

    try:
        if engine.resolved == "intel_gpu" and _is_openvino_repo_model(requested_model_name):
            try:
                logger.info("whisper_openvino_model_try model=%s", requested_model_name)
                text, language_out, duration, segments = _transcribe_with_openvino_model(
                    temp_path=temp_path,
                    model_name=requested_model_name,
                    language=language,
                    prompt=prompt,
                    temperature=temperature,
                )
                logger.info("whisper_openvino_model_ok model=%s", requested_model_name)
                debug_payload = asdict(engine)
                logger.info("whisper_engine=%s", debug_payload)
                return TranscriptionResult(
                    text=text,
                    language=language_out,
                    duration=duration,
                    segments=segments,
                    debug=debug_payload if include_debug else None,
                )
            except Exception as exc:
                fallback_reason = f"{exc.__class__.__name__}:{exc}"
                logger.warning(
                    "whisper_openvino_model_failed fallback_to_faster_whisper model=%s err_type=%s err=%s",
                    requested_model_name,
                    exc.__class__.__name__,
                    exc,
                )
                if require_gpu:
                    raise GpuNotAvailableError(
                        f"{engine.reason};openvino_model_failed:{exc.__class__.__name__}:{exc}"
                    ) from exc
                engine = _fallback_to_faster_whisper_gpu(engine, fallback_reason)
        if engine.resolved == "intel_gpu":
            model = None
            init_errors: List[str] = []
            for gpu_compute_type in _gpu_compute_type_candidates():
                try:
                    logger.info(
                        "whisper_gpu_init_try model=%s compute_type=%s",
                        model_name,
                        gpu_compute_type,
                    )
                    model = _get_model(model_name, engine, compute_type=gpu_compute_type)
                    logger.info(
                        "whisper_gpu_init_ok model=%s compute_type=%s",
                        model_name,
                        gpu_compute_type,
                    )
                    break
                except Exception as exc:
                    init_errors.append(f"{gpu_compute_type}:{exc.__class__.__name__}:{exc}")
                    logger.warning(
                        "whisper_gpu_init_failed model=%s compute_type=%s err=%s",
                        model_name,
                        gpu_compute_type,
                        exc,
                    )
                    if not _is_unsupported_gpu_compute_error(exc):
                        raise
            if model is None:
                reason = f"{engine.reason};model_init_failed:{' | '.join(init_errors)}"
                if require_gpu:
                    raise GpuNotAvailableError(reason)
                logger.warning("whisper_gpu_fallback reason=%s", reason)
                engine = _fallback_to_cpu(engine, reason)
                model = _get_model(model_name, engine)
        else:
            model = _get_model(model_name, engine)
        try:
            segments, info = model.transcribe(
                temp_path,
                language=language,
                initial_prompt=prompt,
                temperature=temperature,
            )
        except Exception as exc:
            if engine.resolved != "intel_gpu":
                raise
            if require_gpu:
                raise GpuNotAvailableError(
                    f"{engine.reason};transcribe_failed:{exc.__class__.__name__}:{exc}"
                ) from exc
            logger.warning(
                "whisper_gpu_transcribe_failed fallback_to_cpu err_type=%s err=%s",
                exc.__class__.__name__,
                exc,
            )
            engine = _fallback_to_cpu(engine, f"transcribe_failed:{exc.__class__.__name__}")
            model = _get_model(model_name, engine)
            segments, info = model.transcribe(
                temp_path,
                language=language,
                initial_prompt=prompt,
                temperature=temperature,
            )

        segment_results: List[SegmentResult] = []
        texts: List[str] = []
        for idx, seg in enumerate(segments):
            cleaned = seg.text.strip()
            if cleaned:
                texts.append(cleaned)
            segment_results.append(
                SegmentResult(
                    id=idx,
                    start=float(seg.start),
                    end=float(seg.end),
                    text=cleaned,
                )
            )

        debug_payload = asdict(engine)
        logger.info("whisper_engine=%s", debug_payload)

        return TranscriptionResult(
            text=" ".join(texts).strip(),
            language=getattr(info, "language", language),
            duration=getattr(info, "duration", None),
            segments=segment_results,
            debug=debug_payload if include_debug else None,
        )
    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)
