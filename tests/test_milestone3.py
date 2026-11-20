"""Unit tesztek a 3. mérföldkő moduljaihoz (GPU és hálózat terhelése nélkül)."""

from __future__ import annotations

from datetime import datetime
import json
from pathlib import Path
import sys
from typing import Any
from unittest.mock import MagicMock

from PIL import Image
from pydantic import ValidationError
import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import audio_processor  # noqa: E402
from database import create_and_populate_db  # noqa: E402
from llm_service import (  # noqa: E402
    StructuredOutputError,
    extract_json_str,
    generate_structured_response,
    run_tourist_pipeline,
    validate_json_response,
)
from schemas import ImageCheckResult, TouristResponse  # noqa: E402
from validator import (  # noqa: E402
    find_place,
    get_open_places_in_category,
    get_places_by_category,
    get_places_by_category_sorted_by_distance,
    is_place_open,
)


@pytest.fixture()
def temp_db_path(tmp_path: Path) -> str:
    """Ideiglenes tourism SQLite adatbázis létrehozása a database.py segítségével."""
    db_file = tmp_path / "test_tourism.db"
    create_and_populate_db(str(db_file))
    return str(db_file)


@pytest.fixture()
def dummy_image() -> Image.Image:
    """Kisméretű tesztkép memóriában a multimodális pipeline tesztekhez."""
    return Image.new("RGB", (64, 64), color=(120, 150, 200))


@pytest.fixture(autouse=True)
def reset_whisper_singleton() -> Any:
    """Minden teszt előtt és után üríti a Whisper modell gyorsítótárat."""
    audio_processor.clear_whisper_cache()
    yield
    audio_processor.clear_whisper_cache()


# ---------------------------------------------------------------------------
# 1. Pydantic JSON validáció és kinyerés
# ---------------------------------------------------------------------------


def test_validate_direct_valid_json() -> None:
    """Közvetlen érvényes JSON helyesen validálódik ImageCheckResult és TouristResponse modellekre."""
    check_raw = json.dumps(
        {
            "is_landmark": True,
            "confidence": 0.94,
            "detected_name": "Országház",
            "reason": "A budapesti Országház neogótikus épülete látható.",
        },
        ensure_ascii=False,
    )
    check_res = validate_json_response(check_raw, ImageCheckResult)
    assert check_res.is_landmark is True
    assert check_res.confidence == pytest.approx(0.94)
    assert check_res.detected_name == "Országház"

    tourist_raw = json.dumps(
        {
            "felismert_hely": "Országház",
            "tortenelem_roviden": "Steindl Imre tervei alapján épült 1885 és 1904 között.",
            "ajanlott_kovetkezo_megallo": "Halászbástya",
            "becsult_ido_percben": 20,
            "megjegyzes": "Érdemes sétálni a Duna-parton.",
        },
        ensure_ascii=False,
    )
    tourist_res = validate_json_response(tourist_raw, TouristResponse)
    assert tourist_res.felismert_hely == "Országház"
    assert tourist_res.ajanlott_kovetkezo_megallo == "Halászbástya"
    assert tourist_res.becsult_ido_percben == 20


@pytest.mark.parametrize(
    "raw_text",
    [
        (
            "Íme az eredmény:\n"
            "```json\n"
            "{\n"
            '  "felismert_hely": "Eiffel-torony",\n'
            '  "tortenelem_roviden": "1889-ben adták át a világkiállításra.",\n'
            '  "ajanlott_kovetkezo_megallo": "Louvre Múzeum",\n'
            '  "becsult_ido_percben": 25,\n'
            '  "megjegyzes": null\n'
            "}\n"
            "```\n"
            "Jó utat kívánok!"
        ),
        (
            "```\n"
            '{"felismert_hely": "Eiffel-torony", '
            '"tortenelem_roviden": "1889-ben adták át.", '
            '"ajanlott_kovetkezo_megallo": "Louvre Múzeum", '
            '"becsult_ido_percben": 25, '
            '"megjegyzes": null}\n'
            "```"
        ),
    ],
)
def test_validate_markdown_fenced_json(raw_text: str) -> None:
    """Markdown kódblokkba (```json és ```) ágyazott JSON sikeresen kinyerhető."""
    parsed = validate_json_response(raw_text, TouristResponse)
    assert parsed.felismert_hely == "Eiffel-torony"
    assert parsed.ajanlott_kovetkezo_megallo == "Louvre Múzeum"
    assert parsed.becsult_ido_percben == 25
    assert parsed.megjegyzes is None


def test_extract_json_from_prose_with_nested_braces() -> None:
    """Extra szöveg közé ágyazott és stringben kapcsos zárójelet tartalmazó JSON kinyerése."""
    raw = (
        "Megjegyzés {nem valid}: "
        '{"is_landmark": true, "confidence": 0.88, '
        '"detected_name": "Pantheon", '
        '"reason": "A {jellegzetes} római kupola látható."} '
        "További szöveg."
    )
    extracted = extract_json_str(raw)
    result = validate_json_response(extracted, ImageCheckResult)
    assert result.detected_name == "Pantheon"
    assert "{jellegzetes}" in result.reason


@pytest.mark.parametrize(
    ("raw_text", "model_cls", "expected_exc"),
    [
        ("Ez egy sima szöveg JSON nélkül.", ImageCheckResult, ValueError),
        ('{"is_landmark": true, "confidence": 0.9', ImageCheckResult, ValueError),
        (
            '{"is_landmark": true, "confidence": 1.5, "detected_name": "X", "reason": "Túl nagy"}',
            ImageCheckResult,
            ValidationError,
        ),
        (
            '{"is_landmark": true, "confidence": -0.1, "detected_name": "X", "reason": "Negatív"}',
            ImageCheckResult,
            ValidationError,
        ),
        (
            '{"felismert_hely": "Országház", "tortenelem_roviden": "Szép"}',
            TouristResponse,
            ValidationError,
        ),
        (
            '{"felismert_hely": "Országház", "tortenelem_roviden": "Szép", '
            '"ajanlott_kovetkezo_megallo": "Halászbástya", "becsult_ido_percben": -5}',
            TouristResponse,
            ValidationError,
        ),
    ],
)
def test_invalid_json_and_schema_validation_errors(
    raw_text: str,
    model_cls: type[ImageCheckResult] | type[TouristResponse],
    expected_exc: type[Exception],
) -> None:
    """Hibás JSON vagy érvénytelen mezőértékek esetén megfelelő kivételt dob."""
    with pytest.raises(expected_exc):
        validate_json_response(raw_text, model_cls)


def test_structured_response_json_retry_succeeds_and_fails() -> None:
    """Hibás JSON után max 1 újrapróbálkozás történik; ha sikerül, visszaadja a modellt."""
    responses = iter(
        [
            "Sajnálom, itt nincs JSON.",
            '{"is_landmark": true, "confidence": 0.91, "detected_name": "Országház", "reason": "OK"}',
        ]
    )
    calls = 0

    def mock_generate(_messages: list[dict[str, Any]]) -> str:
        nonlocal calls
        calls += 1
        return next(responses)

    res = generate_structured_response(
        messages=[{"role": "user", "content": "Teszt"}],
        response_model=ImageCheckResult,
        generate_fn=mock_generate,
        max_json_retries=1,
    )
    assert calls == 2
    assert res.detected_name == "Országház"

    # Mindkétszer hibás JSON -> StructuredOutputError 2 hívás után
    fail_calls = 0

    def always_bad(_messages: list[dict[str, Any]]) -> str:
        nonlocal fail_calls
        fail_calls += 1
        return "hibás kimenet"

    with pytest.raises(StructuredOutputError):
        generate_structured_response(
            messages=[{"role": "user", "content": "Teszt"}],
            response_model=ImageCheckResult,
            generate_fn=always_bad,
            max_json_retries=1,
        )
    assert fail_calls == 2


def test_structured_response_recovers_from_schema_echo() -> None:
    """Ha a modell elsőre a JSON sémát küldi vissza, a retry prompt tiltja a séma-kulcsokat."""
    schema_echo = json.dumps(TouristResponse.model_json_schema(), ensure_ascii=False)
    valid_answer = json.dumps(
        {
            "felismert_hely": "Széchenyi Lánchíd",
            "tortenelem_roviden": "Budapest első állandó kőhídja a Duna felett.",
            "ajanlott_kovetkezo_megallo": "Halászbástya",
            "becsult_ido_percben": 15,
            "megjegyzes": None,
        },
        ensure_ascii=False,
    )
    recorded_prompts: list[list[dict[str, Any]]] = []
    responses = iter([schema_echo, valid_answer])

    def mock_generate(messages: list[dict[str, Any]]) -> str:
        recorded_prompts.append(list(messages))
        return next(responses)

    parsed = generate_structured_response(
        messages=[{"role": "user", "content": "Teszt"}],
        response_model=TouristResponse,
        generate_fn=mock_generate,
        max_json_retries=1,
    )
    assert len(recorded_prompts) == 2
    retry_user_prompt = recorded_prompts[1][-1]["content"]
    assert "properties" in retry_user_prompt
    assert "type" in retry_user_prompt
    assert parsed.felismert_hely == "Széchenyi Lánchíd"
    assert parsed.ajanlott_kovetkezo_megallo == "Halászbástya"


# ---------------------------------------------------------------------------
# 2. Bemeneti guardrail tesztek
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "check_payload",
    [
        {
            "is_landmark": False,
            "confidence": 0.95,
            "detected_name": None,
            "reason": "A képen egy kávéscsésze látható.",
        },
        {
            "is_landmark": True,
            "confidence": 0.45,
            "detected_name": "Ismeretlen torony",
            "reason": "Túl homályos a kép.",
        },
        {
            "is_landmark": True,
            "confidence": 0.59,
            "detected_name": "Országház",
            "reason": "Csak részben látszik az épület.",
        },
    ],
)
def test_guardrail_blocks_non_landmark_or_low_confidence(
    dummy_image: Image.Image,
    temp_db_path: str,
    check_payload: dict[str, Any],
) -> None:
    """Nem-látványosság vagy < 0.6 konfidencia esetén leáll és nem generál útitervet."""
    calls = 0

    def mock_generate(_messages: list[dict[str, Any]]) -> str:
        nonlocal calls
        calls += 1
        if calls == 1:
            return json.dumps(check_payload, ensure_ascii=False)
        raise AssertionError("A guardrail után nem szabad útitervet generálni!")

    result = run_tourist_pipeline(
        image_source=dummy_image,
        db_path=temp_db_path,
        generate_fn=mock_generate,
    )

    assert calls == 1
    assert result.success is False
    assert result.tourist_response is None
    assert result.error_message
    assert result.attempts == 0


def test_guardrail_allows_exact_threshold(
    dummy_image: Image.Image,
    temp_db_path: str,
) -> None:
    """Pontosan 0.6 konfidenciájú látványosság átmegy a guardrail ellenőrzésen."""
    answers = iter(
        [
            json.dumps(
                {
                    "is_landmark": True,
                    "confidence": 0.6,
                    "detected_name": "Országház",
                    "reason": "Elég jól kivehető.",
                },
                ensure_ascii=False,
            ),
            json.dumps(
                {
                    "felismert_hely": "Országház",
                    "tortenelem_roviden": "A magyar Országház Budapest jelképe.",
                    "ajanlott_kovetkezo_megallo": "Halászbástya",
                    "becsult_ido_percben": 15,
                    "megjegyzes": None,
                },
                ensure_ascii=False,
            ),
        ]
    )

    result = run_tourist_pipeline(
        image_source=dummy_image,
        db_path=temp_db_path,
        current_time=datetime(2026, 11, 20, 10, 0),
        generate_fn=lambda _msgs: next(answers),
    )

    assert result.success is True
    assert result.tourist_response is not None
    assert result.tourist_response.ajanlott_kovetkezo_megallo == "Halászbástya"
    assert result.is_next_stop_open is True


# ---------------------------------------------------------------------------
# 3. SQLite lekérdezés és nyitvatartás-ellenőrzés
# ---------------------------------------------------------------------------


def test_sqlite_lookup_and_category_queries(temp_db_path: str) -> None:
    """Helyszínkeresés és kategória-szűrés kis-/nagybetű, ékezet és részleges egyezéssel."""
    assert find_place("Országház", db_path=temp_db_path)["id"] == "parlament"
    assert find_place("parlament", db_path=temp_db_path)["name"] == "Országház"
    assert find_place("ORSZÁGHÁZ", db_path=temp_db_path)["id"] == "parlament"
    assert find_place("Orszaghaz", db_path=temp_db_path)["id"] == "parlament"
    assert find_place("Magyar Parlament", db_path=temp_db_path)["id"] == "parlament"
    assert find_place("Lánchíd", db_path=temp_db_path)["name"] == "Széchenyi Lánchíd"
    assert find_place("Vatikán", db_path=temp_db_path)["id"] == "vatikan"
    assert find_place("Nem létező hely XYZ", db_path=temp_db_path) is None

    budapest_places = get_places_by_category(
        "Budapest",
        db_path=temp_db_path,
        exclude_name="Országház",
    )
    bp_names = {p["name"] for p in budapest_places}
    assert "Országház" not in bp_names
    assert {"Halászbástya", "Széchenyi Lánchíd", "Szent István-bazilika"} <= bp_names

    paris_places = get_places_by_category("PÁRIZS", db_path=temp_db_path)
    assert any(p["name"] == "Eiffel-torony" for p in paris_places)


def test_opening_hours_normal_and_special_cases(temp_db_path: str) -> None:
    """Normál, 00:00-24:00 és 09:00-00:00 nyitvatartások pontos ellenőrzése."""
    # Országház: 08:00-18:00
    open_10, msg_10 = is_place_open(
        "Országház", db_path=temp_db_path, current_time=datetime(2026, 11, 20, 10, 0)
    )
    assert open_10 is True
    assert "nyitva" in msg_10.lower()

    open_08, _ = is_place_open(
        "Országház", db_path=temp_db_path, current_time=datetime(2026, 11, 20, 8, 0)
    )
    assert open_08 is True

    closed_18, _ = is_place_open(
        "Országház", db_path=temp_db_path, current_time=datetime(2026, 11, 20, 18, 0)
    )
    assert closed_18 is False

    closed_20, msg_20 = is_place_open(
        "Országház", db_path=temp_db_path, current_time=datetime(2026, 11, 20, 20, 0)
    )
    assert closed_20 is False
    assert "zárva" in msg_20.lower()

    # Széchenyi Lánchíd: 00:00-24:00 (mindig nyitva)
    for hour, minute in [(0, 0), (3, 30), (12, 0), (23, 59)]:
        is_open, _ = is_place_open(
            "Széchenyi Lánchíd",
            db_path=temp_db_path,
            current_time=datetime(2026, 11, 20, hour, minute),
        )
        assert is_open is True

    # Eiffel-torony: 09:00-00:00 (éjfélig nyitva, ne dobjon ValueError-t!)
    for hour, minute in [(9, 0), (15, 0), (23, 30), (23, 59)]:
        is_open, _ = is_place_open(
            "Eiffel-torony",
            db_path=temp_db_path,
            current_time=datetime(2026, 11, 20, hour, minute),
        )
        assert is_open is True

    for hour, minute in [(0, 0), (4, 0), (8, 59)]:
        is_open, _ = is_place_open(
            "Eiffel-torony",
            db_path=temp_db_path,
            current_time=datetime(2026, 11, 20, hour, minute),
        )
        assert is_open is False

    # Nem létező helyszín
    unknown_open, unknown_msg = is_place_open("Atlantisz", db_path=temp_db_path)
    assert unknown_open is False
    assert "nem található" in unknown_msg.lower()


def test_real_tourism_db_sanity() -> None:
    """Ellenőrzi a repóban lévő tourism.db alapvető lekérdezhetőségét."""
    repo_db = PROJECT_ROOT / "tourism.db"
    if not repo_db.exists():
        pytest.skip("A gyökérkönyvtárban nincs tourism.db")

    is_open, _ = is_place_open(
        "Széchenyi Lánchíd",
        db_path=str(repo_db),
        current_time=datetime(2026, 11, 20, 2, 0),
    )
    assert is_open is True


# ---------------------------------------------------------------------------
# 4. Visszacsatolási hurok (retry loop) mockolt LLM-mel és hanggal
# ---------------------------------------------------------------------------


def test_retry_loop_when_first_suggestion_is_closed(
    dummy_image: Image.Image,
    temp_db_path: str,
) -> None:
    """Zárt első javaslatnál rendszerüzenetet küld a nyitott helyek listájával és újrapróbál."""
    recorded_prompts: list[list[dict[str, Any]]] = []
    # 20:00-kor Budapesten: Szent István-bazilika (09:00-17:45) ZÁRVA,
    # Halászbástya (09:00-23:00) és Széchenyi Lánchíd (00:00-24:00) NYITVA.
    answers = iter(
        [
            json.dumps(
                {
                    "is_landmark": True,
                    "confidence": 0.92,
                    "detected_name": "Országház",
                    "reason": "Országház a Duna partján.",
                },
                ensure_ascii=False,
            ),
            json.dumps(
                {
                    "felismert_hely": "Országház",
                    "tortenelem_roviden": "A magyar parlament épülete.",
                    "ajanlott_kovetkezo_megallo": "Szent István-bazilika",
                    "becsult_ido_percben": 12,
                    "megjegyzes": None,
                },
                ensure_ascii=False,
            ),
            json.dumps(
                {
                    "felismert_hely": "Országház",
                    "tortenelem_roviden": "A magyar parlament épülete.",
                    "ajanlott_kovetkezo_megallo": "Halászbástya",
                    "becsult_ido_percben": 18,
                    "megjegyzes": "Esti panoráma.",
                },
                ensure_ascii=False,
            ),
        ]
    )

    def mock_generate(messages: list[dict[str, Any]]) -> str:
        recorded_prompts.append(list(messages))
        return next(answers)

    result = run_tourist_pipeline(
        image_source=dummy_image,
        db_path=temp_db_path,
        current_time=datetime(2026, 11, 20, 20, 0),
        generate_fn=mock_generate,
        max_closed_retries=2,
    )

    assert len(recorded_prompts) == 3  # 1 image check + 2 itinerary attempts
    assert result.success is True
    assert result.attempts == 2
    assert result.is_next_stop_open is True
    assert result.tourist_response is not None
    assert result.tourist_response.ajanlott_kovetkezo_megallo == "Halászbástya"

    # Ellenőrizzük a visszacsatolási rendszerüzenet tartalmát a 2. útiterv-kérésnél
    feedback_messages = [
        m for m in recorded_prompts[2] if m["role"] == "system" and "jelenleg zárva van" in str(m["content"])
    ]
    assert len(feedback_messages) == 1
    feedback_text = feedback_messages[0]["content"]
    assert "Szent István-bazilika" in feedback_text
    assert "Halászbástya" in feedback_text
    assert "Széchenyi Lánchíd" in feedback_text


def test_retry_loop_exhausts_max_retries(
    dummy_image: Image.Image,
    temp_db_path: str,
) -> None:
    """Ha mind a 3 javaslat zárva van, max 2 újrapróbálkozás után megáll végtelen ciklus nélkül."""
    itinerary_calls = 0

    def mock_generate(messages: list[dict[str, Any]]) -> str:
        nonlocal itinerary_calls
        if any("Vizsgáld meg a képet" in str(m.get("content", "")) for m in messages):
            return json.dumps(
                {
                    "is_landmark": True,
                    "confidence": 0.9,
                    "detected_name": "Országház",
                    "reason": "OK",
                }
            )
        itinerary_calls += 1
        # 21:00-kor mindhárom zárt helyszín Budapesten
        closed_places = [
            "Szent István-bazilika",
            "Széchenyi Gyógyfürdő",
            "Szent István-bazilika",
        ]
        return json.dumps(
            {
                "felismert_hely": "Országház",
                "tortenelem_roviden": "Leírás",
                "ajanlott_kovetkezo_megallo": closed_places[itinerary_calls - 1],
                "becsult_ido_percben": 15,
                "megjegyzes": None,
            },
            ensure_ascii=False,
        )

    result = run_tourist_pipeline(
        image_source=dummy_image,
        db_path=temp_db_path,
        current_time=datetime(2026, 11, 20, 21, 0),
        generate_fn=mock_generate,
        max_closed_retries=2,
    )

    assert itinerary_calls == 3  # 1 kezdeti + max 2 újrapróbálkozás
    assert result.attempts == 3
    assert result.is_next_stop_open is False


def test_pipeline_with_mocked_audio_transcription(
    dummy_image: Image.Image,
    temp_db_path: str,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Hangfájl megadásakor a pipeline beemeli az átiratot a modell kontextusába."""
    audio_file = tmp_path / "kerdes.wav"
    audio_file.write_bytes(b"RIFF\x00\x00\x00\x00WAVEfmt ")

    monkeypatch.setattr(
        "llm_service.transcribe_audio",
        lambda path, device=None: {
            "text": "Hová érdemes továbbmenni a közelben?",
            "language": "hu",
        },
    )

    captured_user_prompts: list[str] = []
    answers = iter(
        [
            json.dumps(
                {
                    "is_landmark": True,
                    "confidence": 0.95,
                    "detected_name": "Országház",
                    "reason": "OK",
                }
            ),
            json.dumps(
                {
                    "felismert_hely": "Országház",
                    "tortenelem_roviden": "Budapest egyik fő nevezetessége.",
                    "ajanlott_kovetkezo_megallo": "Széchenyi Lánchíd",
                    "becsult_ido_percben": 10,
                    "megjegyzes": "Sétával könnyen elérhető.",
                },
                ensure_ascii=False,
            ),
        ]
    )

    def mock_generate(messages: list[dict[str, Any]]) -> str:
        for msg in messages:
            if msg["role"] == "user" and isinstance(msg["content"], list):
                for part in msg["content"]:
                    if part.get("type") == "text":
                        captured_user_prompts.append(part["text"])
        return next(answers)

    result = run_tourist_pipeline(
        image_source=dummy_image,
        audio_path=str(audio_file),
        db_path=temp_db_path,
        current_time=datetime(2026, 11, 20, 11, 0),
        whisper_device="cpu",
        generate_fn=mock_generate,
    )

    assert result.success is True
    assert result.transcript == {
        "text": "Hová érdemes továbbmenni a közelben?",
        "language": "hu",
    }
    assert any("Hová érdemes továbbmenni a közelben?" in p for p in captured_user_prompts)


# ---------------------------------------------------------------------------
# 5. Hangfeldolgozás (audio_processor.py) tesztek mockolt Whisperrel
# ---------------------------------------------------------------------------


def test_audio_file_validation_errors(tmp_path: Path) -> None:
    """Hiányzó, üres és nem támogatott hangfájlokra tiszta kivételeket dob."""
    with pytest.raises(audio_processor.AudioFileNotFoundError):
        audio_processor.transcribe_audio(str(tmp_path / "nem_letezik.wav"))

    empty_wav = tmp_path / "ures.wav"
    empty_wav.write_bytes(b"")
    with pytest.raises(audio_processor.EmptyAudioFileError):
        audio_processor.transcribe_audio(str(empty_wav))

    bad_ext = tmp_path / "szoveg.txt"
    bad_ext.write_text("nem hang", encoding="utf-8")
    with pytest.raises(audio_processor.UnsupportedAudioFormatError):
        audio_processor.transcribe_audio(str(bad_ext))


def test_audio_ffmpeg_errors(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Hiányzó vagy hibázó FFmpeg esetén FFmpegError kivétel keletkezik."""
    sample_wav = tmp_path / "sample.wav"
    sample_wav.write_bytes(b"RIFFdata")

    monkeypatch.setattr(audio_processor.shutil, "which", lambda _cmd: None)
    with pytest.raises(audio_processor.FFmpegError):
        audio_processor.transcribe_audio(str(sample_wav), device="cpu")

    # FFmpeg jelen van, de a dekódolás közben RuntimeError-t dob
    monkeypatch.setattr(audio_processor.shutil, "which", lambda _cmd: "C:/ffmpeg/ffmpeg.exe")
    fake_model = MagicMock()
    fake_model.transcribe.side_effect = RuntimeError("Failed to load audio: ffmpeg error")
    fake_whisper = MagicMock()
    fake_whisper.load_model.return_value = fake_model
    monkeypatch.setattr(audio_processor, "whisper", fake_whisper)

    with pytest.raises(audio_processor.FFmpegError):
        audio_processor.transcribe_audio(str(sample_wav), device="cpu")


def test_audio_singleton_cache_cpu_and_language_modes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Whisper modell cache/singleton működése, CPU fp16=False és nyelvi módok ellenőrzése."""
    sample_mp3 = tmp_path / "hang.mp3"
    sample_mp3.write_bytes(b"ID3dummydata")

    monkeypatch.setattr(audio_processor.shutil, "which", lambda _cmd: "ffmpeg")
    fake_model = MagicMock()
    fake_model.transcribe.side_effect = [
        {"text": "  Szia Budapest!  ", "language": "hu"},
        {"text": "  Hello Paris! ", "language": "en"},
    ]
    fake_whisper = MagicMock()
    fake_whisper.load_model.return_value = fake_model
    monkeypatch.setattr(audio_processor, "whisper", fake_whisper)

    # 1. hívás: magyarra optimalizált mód, CPU-n (fp16=False)
    res_hu = audio_processor.transcribe_audio(
        str(sample_mp3),
        language="hu",
        device="cpu",
    )
    assert res_hu == {"text": "Szia Budapest!", "language": "hu"}
    first_kwargs = fake_model.transcribe.call_args_list[0].kwargs
    assert first_kwargs["fp16"] is False
    assert first_kwargs["language"] == "hu"
    assert "initial_prompt" in first_kwargs

    # 2. hívás: automatikus nyelvdetektálás (language=None), ugyanazzal a cached modellel
    res_auto = audio_processor.transcribe_audio(
        str(sample_mp3),
        language=None,
        device="cpu",
    )
    assert res_auto == {"text": "Hello Paris!", "language": "en"}
    second_kwargs = fake_model.transcribe.call_args_list[1].kwargs
    assert second_kwargs["fp16"] is False
    assert "language" not in second_kwargs

    # A small modell csak egyszer töltődött be (singleton cache)
    fake_whisper.load_model.assert_called_once_with("small", device="cpu")


# ---------------------------------------------------------------------------
# 6. Távolság-első, nyitvatartás-második ajánlási logika tesztek
# ---------------------------------------------------------------------------


def test_distance_first_sorting_from_orszaghaz(temp_db_path: str) -> None:
    """Az Országháztól indulva szigorúan távolság szerint növekvő sorrendben rendez."""
    orszaghaz = find_place("Országház", db_path=temp_db_path)
    assert orszaghaz is not None

    sorted_places = get_places_by_category_sorted_by_distance(
        "budapest",
        origin_lat=orszaghaz["lat"],
        origin_lon=orszaghaz["lon"],
        db_path=temp_db_path,
        exclude_id=orszaghaz["id"],
    )
    assert [p["name"] for p in sorted_places] == [
        "Szent István-bazilika",
        "Széchenyi Lánchíd",
        "Halászbástya",
        "Széchenyi Gyógyfürdő",
    ]
    assert sorted_places[0]["distance_km"] < sorted_places[1]["distance_km"] < sorted_places[2]["distance_km"]

    # 20:00-kor a nyitva tartó helyek között is a Széchenyi Lánchíd van elöl
    open_at_20 = get_open_places_in_category(
        "budapest",
        db_path=temp_db_path,
        current_time=datetime(2026, 11, 20, 20, 0),
        exclude_names=[orszaghaz["name"]],
        origin_coords=(orszaghaz["lat"], orszaghaz["lon"]),
    )
    assert [p["name"] for p in open_at_20] == [
        "Széchenyi Lánchíd",
        "Halászbástya",
    ]


def test_pipeline_distance_first_then_opening_hours(
    dummy_image: Image.Image,
    temp_db_path: str,
) -> None:
    """Először távolság alapján választ (Szent István-bazilika), majd nyitvatartás alapján a Lánchídra vált."""
    recorded_prompts: list[list[dict[str, Any]]] = []
    answers = iter(
        [
            json.dumps(
                {
                    "is_landmark": True,
                    "confidence": 0.96,
                    "detected_name": "Országház",
                    "reason": "Az Országház látható a képen.",
                },
                ensure_ascii=False,
            ),
            json.dumps(
                {
                    "felismert_hely": "Országház",
                    "tortenelem_roviden": "A magyar parlament épülete a Duna partján.",
                    "ajanlott_kovetkezo_megallo": "Szent István-bazilika",
                    "becsult_ido_percben": 11,
                    "megjegyzes": None,
                },
                ensure_ascii=False,
            ),
            json.dumps(
                {
                    "felismert_hely": "Országház",
                    "tortenelem_roviden": "A magyar parlament épülete a Duna partján.",
                    "ajanlott_kovetkezo_megallo": "Széchenyi Lánchíd",
                    "becsult_ido_percben": 12,
                    "megjegyzes": "0-24 nyitva tartó ikonikus híd.",
                },
                ensure_ascii=False,
            ),
        ]
    )

    def mock_generate(messages: list[dict[str, Any]]) -> str:
        recorded_prompts.append(list(messages))
        return next(answers)

    result = run_tourist_pipeline(
        image_source=dummy_image,
        db_path=temp_db_path,
        current_time=datetime(2026, 11, 20, 20, 0),
        generate_fn=mock_generate,
        max_closed_retries=2,
    )

    # Ellenőrizzük, hogy az első útiterv-promptban távolság szerinti a lista és nincs NYITVA/ZÁRVA előszűrés
    initial_user_text = next(
        part["text"]
        for part in recorded_prompts[1][1]["content"]
        if part.get("type") == "text"
    )
    assert "NYITVA" not in initial_user_text
    assert "ZÁRVA" not in initial_user_text
    idx_bazilika = initial_user_text.index("1. Szent István-bazilika")
    idx_lanchid = initial_user_text.index("2. Széchenyi Lánchíd")
    idx_halasz = initial_user_text.index("3. Halászbástya")
    idx_furdo = initial_user_text.index("4. Széchenyi Gyógyfürdő")
    assert idx_bazilika < idx_lanchid < idx_halasz < idx_furdo

    # A visszacsatolásban a Széchenyi Lánchíd előzze meg a Halászbástyát
    feedback_msg = next(
        m["content"]
        for m in recorded_prompts[2]
        if m["role"] == "system" and "jelenleg zárva van" in str(m["content"])
    )
    assert feedback_msg.index("Széchenyi Lánchíd") < feedback_msg.index("Halászbástya")

    assert result.success is True
    assert result.attempts == 2
    assert result.is_next_stop_open is True
    assert result.tourist_response is not None
    assert result.tourist_response.ajanlott_kovetkezo_megallo == "Széchenyi Lánchíd"
    assert result.next_stop_distance_km == pytest.approx(1.034, abs=0.01)
