"""Hangfeldolgozó modul lokális Whisper (STT) integrációval."""

from __future__ import annotations

import logging
import os
from pathlib import Path
import shutil
import subprocess
from typing import Any
import warnings

logging.getLogger("torch.utils.flop_counter").setLevel(logging.ERROR)
import torch

try:
    import whisper
except ImportError:  # pragma: no cover
    whisper = None

SUPPORTED_AUDIO_EXTENSIONS: frozenset[str] = frozenset({".wav", ".mp3", ".m4a"})
DEFAULT_WHISPER_MODEL: str = "tiny"
HUNGARIAN_TOURISM_PROMPT: str = (
    "Turisztikai asszisztens, nevezetességek: Országház, Halászbástya, "
    "Széchenyi Lánchíd, Szent István-bazilika, Széchenyi Gyógyfürdő, "
    "Eiffel-torony, Louvre Múzeum, Notre-Dame, Trevi-kút, Colosseum."
)

_MODEL_CACHE: dict[tuple[str, str], Any] = {}


class AudioProcessingError(Exception):
    """Alaposztály a hangfeldolgozás során felmerülő hibákhoz."""


class AudioFileNotFoundError(AudioProcessingError, FileNotFoundError):
    """A megadott hangfájl nem található vagy nem érvényes fájl."""


class EmptyAudioFileError(AudioProcessingError, ValueError):
    """A megadott hangfájl üres (0 bájt méretű)."""


class UnsupportedAudioFormatError(AudioProcessingError, ValueError):
    """A hangfájl formátuma nem támogatott."""


class FFmpegError(AudioProcessingError, RuntimeError):
    """FFmpeg hiányából vagy futási hibájából eredő kivétel."""


def resolve_device(device: str | None = None) -> str:
    """Meghatározza a Whisper futtatásához használt eszközt ('cuda' vagy 'cpu')."""
    if device is None:
        env_device = os.environ.get("WHISPER_DEVICE")
        if env_device and env_device.strip():
            return resolve_device(env_device)
        return "cuda" if torch.cuda.is_available() else "cpu"

    normalized = device.strip().lower()
    if normalized == "cpu":
        return "cpu"
    if normalized == "cuda" or normalized.startswith("cuda:"):
        if not torch.cuda.is_available():
            raise AudioProcessingError(
                "CUDA eszköz lett megadva a Whisperhez, de GPU nem érhető el."
            )
        return normalized
    raise ValueError(
        f"Érvénytelen futtatási eszköz: {device!r}. Használd a 'cpu' vagy 'cuda' értéket."
    )


def _get_whisper_model_with_device(
    model_name: str = DEFAULT_WHISPER_MODEL,
    device: str | None = None,
) -> tuple[Any, str]:
    """Betölti és gyorsítótárazza a Whisper modellt, szükség esetén CPU-ra váltva."""
    if whisper is None:
        raise AudioProcessingError(
            "Az 'openai-whisper' csomag nincs telepítve a környezetben."
        )

    resolved_device = resolve_device(device)
    cache_key = (model_name, resolved_device)
    if cache_key in _MODEL_CACHE:
        return _MODEL_CACHE[cache_key], resolved_device

    try:
        model = whisper.load_model(model_name, device=resolved_device)
    except RuntimeError as exc:
        is_oom = isinstance(exc, torch.cuda.OutOfMemoryError) or (
            "out of memory" in str(exc).lower()
        )
        if device is None and resolved_device.startswith("cuda") and is_oom:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            resolved_device = "cpu"
            cache_key = (model_name, resolved_device)
            if cache_key in _MODEL_CACHE:
                return _MODEL_CACHE[cache_key], resolved_device
            model = whisper.load_model(model_name, device=resolved_device)
        else:
            raise AudioProcessingError(
                f"Nem sikerült betölteni a Whisper ({model_name}) modellt: {exc}"
            ) from exc

    _MODEL_CACHE[cache_key] = model
    return model, resolved_device


def get_whisper_model(
    model_name: str = DEFAULT_WHISPER_MODEL,
    device: str | None = None,
) -> Any:
    """Visszaadja a gyorsítótárazott (singleton) Whisper modellt."""
    model, _ = _get_whisper_model_with_device(model_name=model_name, device=device)
    return model


def clear_whisper_cache() -> None:
    """Kiüríti a gyorsítótárazott Whisper modelleket és felszabadítja a CUDA memóriát."""
    _MODEL_CACHE.clear()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def _validate_audio_file(file_path: str) -> Path:
    """Ellenőrzi a hangfájl létezését, kiterjesztését és nem üres méretét."""
    if not file_path or not isinstance(file_path, str) or not file_path.strip():
        raise AudioFileNotFoundError("A hangfájl elérési útja nem lehet üres.")

    path = Path(file_path)
    if not path.exists() or not path.is_file():
        raise AudioFileNotFoundError(f"A hangfájl nem található: {file_path}")

    suffix = path.suffix.lower()
    if suffix not in SUPPORTED_AUDIO_EXTENSIONS:
        supported = ", ".join(sorted(SUPPORTED_AUDIO_EXTENSIONS))
        raise UnsupportedAudioFormatError(
            f"Nem támogatott hangformátum ({suffix or 'nincs kiterjesztés'}). "
            f"Támogatott formátumok: {supported}"
        )

    if path.stat().st_size == 0:
        raise EmptyAudioFileError(f"A hangfájl üres (0 bájt): {file_path}")

    return path


def transcribe_audio(
    file_path: str,
    language: str | None = None,
    device: str | None = None,
    model_name: str = DEFAULT_WHISPER_MODEL,
    initial_prompt: str | None = None,
) -> dict[str, str]:
    """Hangfájl (.wav, .mp3, .m4a) átírása szöveggé lokális Whisper modellel.

    Args:
        file_path: A feldolgozandó hangfájl elérési útja.
        language: Opcionális nyelvkód (pl. "hu"); None esetén automatikus nyelvdetektálás.
        device: Futtatási eszköz ("cpu" vagy "cuda"); None esetén automatikus választás.
        model_name: A betöltendő Whisper modell neve (alapértelmezett: "small").
        initial_prompt: Opcionális kontextusprompt a felismerés pontosságához.

    Returns:
        Szótár a felismert szöveggel ("text") és a detektált nyelvvel ("language").
    """
    path = _validate_audio_file(file_path)

    if shutil.which("ffmpeg") is None:
        raise FFmpegError(
            "Az FFmpeg nem található a rendszer PATH változójában. "
            "Kérlek telepítsd az FFmpeg-et a hangfájlok feldolgozásához."
        )

    model, resolved_device = _get_whisper_model_with_device(
        model_name=model_name,
        device=device,
    )

    options: dict[str, Any] = {
        "task": "transcribe",
        "fp16": resolved_device.startswith("cuda"),
    }

    if language is not None:
        options["language"] = language
        if language.strip().lower() in ("hu", "hungarian") and initial_prompt is None:
            options["initial_prompt"] = HUNGARIAN_TOURISM_PROMPT

    if initial_prompt is not None:
        options["initial_prompt"] = initial_prompt

    try:
        with warnings.catch_warnings():
            if resolved_device == "cpu":
                warnings.filterwarnings(
                    "ignore",
                    message=r"Performing inference on CPU when CUDA is available",
                    category=UserWarning,
                )
            result = model.transcribe(str(path), **options)
    except (FileNotFoundError, subprocess.CalledProcessError) as exc:
        raise FFmpegError(
            f"FFmpeg hiba a hangfájl dekódolása közben ({path.name}): {exc}"
        ) from exc
    except RuntimeError as exc:
        msg_lower = str(exc).lower()
        if "ffmpeg" in msg_lower or "failed to load audio" in msg_lower:
            raise FFmpegError(
                f"FFmpeg hiba a hangfájl betöltése közben ({path.name}): {exc}"
            ) from exc
        raise AudioProcessingError(
            f"Futási hiba a hangfájl átírása során ({path.name}): {exc}"
        ) from exc
    except AudioProcessingError:
        raise
    except Exception as exc:
        raise AudioProcessingError(
            f"Váratlan hiba a hangfeldolgozás során ({path.name}): {exc}"
        ) from exc

    detected_language = str(result.get("language") or language or "unknown").strip()
    transcript_text = str(result.get("text", "")).strip()

    return {
        "text": transcript_text,
        "language": detected_language,
    }
