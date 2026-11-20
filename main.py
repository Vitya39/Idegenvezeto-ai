"""Parancssori belépési pont a multimodális turisztikai asszisztenshez."""

from __future__ import annotations

import argparse
from datetime import datetime
import json
import logging
import sys

logging.getLogger("torch.utils.flop_counter").setLevel(logging.ERROR)

from audio_processor import AudioProcessingError
from llm_service import StructuredOutputError, run_tourist_pipeline
from validator import DEFAULT_DB_PATH


def parse_args() -> argparse.Namespace:
    """Feldolgozza a parancssori argumentumokat."""
    parser = argparse.ArgumentParser(
        description="Helyi multimodális turisztikai asszisztens (Ollama nélkül).",
    )
    parser.add_argument(
        "--image",
        default="kepek/test2.jpg",
        help="A felismerendő látványosságot tartalmazó kép elérési útja.",
    )
    parser.add_argument(
        "--audio",
        default="hang/Recording.mp3",
        help="Opcionális hangüzenet fájl (.wav, .mp3, .m4a).",
    )
    parser.add_argument(
        "--text",
        default=None,
        help="Opcionális szöveges kérdés vagy megjegyzés a turistától.",
    )
    parser.add_argument(
        "--whisper-device",
        default=None,
        help="Whisper futtatási eszköz ('cpu' vagy 'cuda').",
    )
    parser.add_argument(
        "--time",
        default=None,
        help="Opcionális tesztidőpont HH:MM formátumban (pl. 20:00).",
    )
    parser.add_argument(
        "--db",
        default=DEFAULT_DB_PATH,
        help="Az SQLite adatbázis elérési útja.",
    )
    return parser.parse_args()


def _parse_time_override(time_str: str | None) -> datetime | None:
    """HH:MM időbélyeget mai datetime objektummá alakít."""
    if not time_str:
        return None
    parsed = datetime.strptime(time_str.strip(), "%H:%M")
    return datetime.now().replace(
        hour=parsed.hour,
        minute=parsed.minute,
        second=0,
        microsecond=0,
    )


def main() -> int:
    """Elindítja a teljes turisztikai pipeline-t és JSON formátumban kiírja az eredményt."""
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    if hasattr(sys.stderr, "reconfigure"):
        sys.stderr.reconfigure(encoding="utf-8")

    args = parse_args()
    try:
        current_time = _parse_time_override(args.time)
        result = run_tourist_pipeline(
            image_source=args.image,
            audio_path=args.audio or None,
            user_text=args.text,
            db_path=args.db,
            current_time=current_time,
            whisper_device=args.whisper_device,
        )
    except (
        AudioProcessingError,
        StructuredOutputError,
        FileNotFoundError,
        ValueError,
    ) as exc:
        error_payload = {
            "success": False,
            "error_type": type(exc).__name__,
            "error_message": str(exc),
        }
        print(json.dumps(error_payload, ensure_ascii=False, indent=2), file=sys.stderr)
        return 1

    print(json.dumps(result.model_dump(), ensure_ascii=False, indent=2))
    return 0 if result.success else 1


if __name__ == "__main__":
    raise SystemExit(main())
