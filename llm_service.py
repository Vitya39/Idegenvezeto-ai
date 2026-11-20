"""Lokális Qwen2.5-VL-3B-Instruct LLM szolgáltatás, JSON-validáció és pipeline logika."""

from __future__ import annotations

from collections.abc import Callable
from datetime import datetime
import json
import logging
from pathlib import Path
import re
from typing import Any, TypeVar

from PIL import Image
from pydantic import BaseModel, ValidationError

from audio_processor import transcribe_audio
from location import get_image_coordinates
from schemas import ImageCheckResult, PipelineResult, TouristResponse
from validator import (
    DEFAULT_DB_PATH,
    build_closed_place_feedback,
    find_place,
    get_all_places,
    get_open_places_in_category,
    get_places_by_category,
    get_places_by_category_sorted_by_distance,
    haversine_distance,
    is_place_open,
)

VISION_MODEL_ID: str = "Qwen/Qwen2.5-VL-3B-Instruct"
MIN_PIXELS: int = 256 * 28 * 28
MAX_PIXELS: int = 512 * 28 * 28
LANDMARK_CONFIDENCE_THRESHOLD: float = 0.6
DEFAULT_MAX_JSON_RETRIES: int = 1
DEFAULT_MAX_CLOSED_RETRIES: int = 2

T = TypeVar("T", bound=BaseModel)

_CODE_FENCE_PATTERN = re.compile(
    r"```(?:json)?\s*(.*?)\s*```",
    flags=re.DOTALL | re.IGNORECASE,
)

_qwen_model: Any = None
_qwen_processor: Any = None
_loaded_model_config: tuple[str, bool] | None = None
_default_llm_service: LocalQwenVLService | None = None


class StructuredOutputError(ValueError):
    """Kivétel, ha a modell több próbálkozás után sem adott érvényes JSON választ."""

    def __init__(
        self,
        message: str,
        raw_output: str = "",
        last_error: Exception | None = None,
    ) -> None:
        super().__init__(message)
        self.raw_output = raw_output
        self.last_error = last_error


def clear_qwen_cache() -> None:
    """Felszabadítja a betöltött Qwen2.5-VL modellt és a CUDA gyorsítótárat."""
    global _qwen_model, _qwen_processor, _loaded_model_config
    _qwen_model = None
    _qwen_processor = None
    _loaded_model_config = None
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except ImportError:  # pragma: no cover
        pass


def load_qwen_vl_model(
    model_id: str = VISION_MODEL_ID,
    use_4bit: bool = True,
) -> tuple[Any, Any]:
    """Lustán betölti és singletonként gyorsítótárazza a Qwen2.5-VL-3B-Instruct modellt."""
    global _qwen_model, _qwen_processor, _loaded_model_config

    if (
        _qwen_model is not None
        and _qwen_processor is not None
        and _loaded_model_config == (model_id, use_4bit)
    ):
        return _qwen_model, _qwen_processor

    if _qwen_model is not None:
        clear_qwen_cache()

    import warnings
    logging.getLogger("torch.utils.flop_counter").setLevel(logging.ERROR)
    import torch
    from transformers import (
        AutoProcessor,
        BitsAndBytesConfig,
        Qwen2_5_VLForConditionalGeneration,
        logging as hf_logging,
    )

    warnings.filterwarnings("ignore", category=UserWarning)
    hf_logging.set_verbosity_error()

    has_cuda = torch.cuda.is_available()
    if has_cuda:
        use_bf16 = (not use_4bit) and torch.cuda.is_bf16_supported()
        model_dtype = torch.bfloat16 if use_bf16 else torch.float16
    else:
        model_dtype = torch.float32

    bnb_config = (
        BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
        )
        if has_cuda and use_4bit
        else None
    )

    _qwen_model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        model_id,
        quantization_config=bnb_config,
        dtype=model_dtype,
        device_map="auto" if has_cuda else None,
        low_cpu_mem_usage=True,
    )
    _qwen_processor = AutoProcessor.from_pretrained(
        model_id,
        size={"shortest_edge": MIN_PIXELS, "longest_edge": MAX_PIXELS},
    )
    _loaded_model_config = (model_id, use_4bit)
    return _qwen_model, _qwen_processor


class LocalQwenVLService:
    """Lokális Qwen2.5-VL-3B-Instruct modellhívásokat kezelő szolgáltatás (Ollama nélkül)."""

    def __init__(
        self,
        model_id: str = VISION_MODEL_ID,
        use_4bit: bool = True,
    ) -> None:
        self.model_id = model_id
        self.use_4bit = use_4bit

    def generate(
        self,
        messages: list[dict[str, Any]],
        max_new_tokens: int = 512,
    ) -> str:
        """Szöveges vagy multimodális választ generál a lokális Qwen2.5-VL modellel."""
        from qwen_vl_utils import process_vision_info
        import torch

        model, processor = load_qwen_vl_model(
            model_id=self.model_id,
            use_4bit=self.use_4bit,
        )
        text = processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        image_inputs, video_inputs = process_vision_info(messages)

        target_device = "cuda" if torch.cuda.is_available() else "cpu"
        inputs = processor(
            text=[text],
            images=image_inputs or None,
            videos=video_inputs or None,
            padding=True,
            return_tensors="pt",
        ).to(target_device)

        with torch.inference_mode():
            generated_ids = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
            )

        generated_ids_trimmed = [
            out_ids[len(in_ids) :]
            for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        return processor.batch_decode(
            generated_ids_trimmed,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )[0].strip()


def get_default_llm_service() -> LocalQwenVLService:
    """Visszaadja az alapértelmezett singleton LocalQwenVLService példányt."""
    global _default_llm_service
    if _default_llm_service is None:
        _default_llm_service = LocalQwenVLService()
    return _default_llm_service


def _extract_balanced_json_candidates(text: str) -> list[str]:
    """Kigyűjti a szövegben található, zárójel-egyensúlyban lévő {...} JSON jelölteket."""
    candidates: list[str] = []
    idx = 0
    length = len(text)

    while idx < length:
        if text[idx] != "{":
            idx += 1
            continue

        start = idx
        depth = 0
        in_string = False
        escape = False
        pos = start
        found_close = False

        while pos < length:
            ch = text[pos]
            if in_string:
                if escape:
                    escape = False
                elif ch == "\\":
                    escape = True
                elif ch == '"':
                    in_string = False
            else:
                if ch == '"':
                    in_string = True
                elif ch == "{":
                    depth += 1
                elif ch == "}":
                    depth -= 1
                    if depth == 0:
                        candidates.append(text[start : pos + 1])
                        idx = pos + 1
                        found_close = True
                        break
            pos += 1

        if not found_close:
            idx = start + 1

    return candidates


def extract_json_str(raw_text: str) -> str:
    """Robusztusan kinyeri a JSON objektumot markdown blokkok vagy extra szöveg közül."""
    if not raw_text or not raw_text.strip():
        raise ValueError("Üres válasz érkezett a modelltől.")

    fallback_candidate: str | None = None

    for block in _CODE_FENCE_PATTERN.findall(raw_text):
        for candidate in _extract_balanced_json_candidates(block):
            if fallback_candidate is None:
                fallback_candidate = candidate
            try:
                json.loads(candidate)
                return candidate
            except json.JSONDecodeError:
                continue

    for candidate in _extract_balanced_json_candidates(raw_text):
        if fallback_candidate is None:
            fallback_candidate = candidate
        try:
            json.loads(candidate)
            return candidate
        except json.JSONDecodeError:
            continue

    if fallback_candidate is not None:
        return fallback_candidate

    raise ValueError("Nem található érvényes JSON objektum a modell válaszában.")


def validate_json_response(raw_text: str, response_model: type[T]) -> T:
    """Kinyeri a JSON-t a nyers szövegből és validálja a megadott Pydantic modellel."""
    json_str = extract_json_str(raw_text)
    return response_model.model_validate_json(json_str)


def _build_json_example_template(response_model: type[BaseModel]) -> str:
    """Konkrét, kitöltendő JSON példaobjektumot készít a Pydantic modell mezőiből."""
    schema = response_model.model_json_schema()
    properties = schema.get("properties", {})
    example: dict[str, Any] = {}

    for field_name, field_info in properties.items():
        description = field_info.get("description") or field_name
        field_type = field_info.get("type")
        if not field_type and "anyOf" in field_info:
            non_null_types = [
                item.get("type")
                for item in field_info["anyOf"]
                if isinstance(item, dict) and item.get("type") != "null"
            ]
            field_type = non_null_types[0] if non_null_types else None

        if field_type == "boolean":
            example[field_name] = True
        elif field_type == "integer":
            example[field_name] = 15
        elif field_type == "number":
            example[field_name] = 0.9
        elif field_type == "string":
            example[field_name] = f"<{description}>"
        else:
            example[field_name] = None

    return json.dumps(example, ensure_ascii=False, indent=2)


def build_json_system_prompt(
    response_model: type[BaseModel],
    task_instruction: str,
) -> str:
    """Összeállítja a Pydantic model_json_schema() alapú szigorú JSON system promptot."""
    schema_json = json.dumps(
        response_model.model_json_schema(),
        ensure_ascii=False,
    )
    example_json = _build_json_example_template(response_model)
    return (
        f"{task_instruction}\n\n"
        "FONTOS: Kizárólag egyetlen kitöltött, érvényes JSON objektummal válaszolj, "
        "minden egyéb magyarázó szöveg és markdown kódblokk nélkül!\n"
        "SOHA ne a JSON séma leírását ('properties', 'required', 'title', 'type' kulcsokat) "
        "add vissza, hanem magukat a kitöltött adatmezőket az alábbi minta formátumában:\n"
        f"{example_json}\n\n"
        f"Referencia JSON séma (model_json_schema): {schema_json}"
    )


def generate_structured_response(
    messages: list[dict[str, Any]],
    response_model: type[T],
    generate_fn: Callable[[list[dict[str, Any]]], str] | None = None,
    max_json_retries: int = DEFAULT_MAX_JSON_RETRIES,
    max_new_tokens: int = 512,
) -> T:
    """Meghívja a modellt és validálja a JSON választ, hibás JSON esetén újrapróbálja."""
    current_messages = list(messages)
    caller = generate_fn or (
        lambda msgs: get_default_llm_service().generate(
            msgs, max_new_tokens=max_new_tokens
        )
    )
    example_json = _build_json_example_template(response_model)

    for attempt in range(max_json_retries + 1):
        raw_output = caller(current_messages)
        try:
            return validate_json_response(raw_output, response_model)
        except (ValueError, ValidationError) as exc:
            if attempt < max_json_retries:
                is_schema_echo = (
                    '"properties"' in raw_output and '"type"' in raw_output
                )
                extra_hint = (
                    "HIBA: A JSON séma definícióját küldted vissza a kitöltött válasz helyett! "
                    "Ne használd a 'properties', 'required', 'title', 'type' kulcsokat! "
                    if is_schema_echo
                    else ""
                )
                current_messages.append({"role": "assistant", "content": raw_output})
                current_messages.append(
                    {
                        "role": "user",
                        "content": (
                            f"{extra_hint}"
                            f"A válaszod nem érvényes JSON vagy nem felel meg az elvárt mezőknek ({exc}). "
                            "Kérlek, kizárólag egyetlen kitöltött érvényes JSON objektumot adj vissza "
                            "pontosan ezekkel a kulcsokkal, minden egyéb szöveg nélkül:\n"
                            f"{example_json}"
                        ),
                    }
                )
            else:
                raise StructuredOutputError(
                    f"A modell válasza {max_json_retries + 1} próbálkozás után sem érvényes "
                    f"{response_model.__name__} JSON: {exc}",
                    raw_output=raw_output,
                    last_error=exc,
                ) from exc

    raise StructuredOutputError("Váratlan hiba a strukturált válasz generálása közben.")


def evaluate_image_guardrail(
    check_result: ImageCheckResult,
    min_confidence: float = LANDMARK_CONFIDENCE_THRESHOLD,
) -> tuple[bool, str]:
    """Bemeneti guardrail: ellenőrzi, hogy a kép megbízhatóan ábrázol-e látványosságot."""
    if not check_result.is_landmark:
        return (
            False,
            "A feltöltött képen nem ismerhető fel egyértelmű turisztikai látványosság "
            f"({check_result.reason}). Kérlek küldj egy jól látható nevezetességről készült fotót!",
        )
    if check_result.confidence < min_confidence:
        return (
            False,
            f"A látványosság felismerési megbízhatósága túl alacsony "
            f"({check_result.confidence:.0%} < {min_confidence:.0%}). "
            "Kérlek készíts élesebb, közelebbi fotót a nevezetességről!",
        )
    return True, "OK"


def _open_rgb_image(image_source: str | Path | Image.Image) -> Image.Image:
    """Megnyitja és RGB formátumra konvertálja a bemeneti képet fájlzárás nélkül."""
    if isinstance(image_source, (str, Path)):
        with Image.open(image_source) as img:
            return img.convert("RGB")
    return image_source.convert("RGB")


def check_image_landmark(
    image_source: str | Path | Image.Image,
    generate_fn: Callable[[list[dict[str, Any]]], str] | None = None,
    max_json_retries: int = DEFAULT_MAX_JSON_RETRIES,
    known_places: list[str] | None = None,
) -> ImageCheckResult:
    """Ellenőrzi a Qwen2.5-VL modellel, hogy a képen látványosság szerepel-e."""
    image_pil = _open_rgb_image(image_source)
    system_prompt = build_json_system_prompt(
        ImageCheckResult,
        "Te egy vizuális látványosság-ellenőrző modul vagy. Elemezd a képet, és döntsd el, "
        "hogy valódi turisztikai látványosság vagy nevezetesség látható-e rajta.",
    )
    places_hint = ""
    if known_places:
        places_hint = (
            "\nHa a képen az alábbi ismert nevezetességek egyike látható, "
            "a detected_name mezőben lehetőség szerint a pontos nevét add meg: "
            + ", ".join(known_places)
            + " (ha viszont nem látványosság látható a képen, az is_landmark legyen false)."
        )
    messages: list[dict[str, Any]] = [
        {"role": "system", "content": system_prompt},
        {
            "role": "user",
            "content": [
                {
                    "type": "image",
                    "image": image_pil,
                    "min_pixels": MIN_PIXELS,
                    "max_pixels": MAX_PIXELS,
                },
                {
                    "type": "text",
                    "text": (
                        "Vizsgáld meg a képet! Add meg az is_landmark (bool), "
                        "a confidence (0.0 és 1.0 között), a detected_name "
                        "(a látványosság neve vagy null) és a reason (rövid magyar indoklás) mezőket."
                        f"{places_hint}"
                    ),
                },
            ],
        },
    ]
    return generate_structured_response(
        messages=messages,
        response_model=ImageCheckResult,
        generate_fn=generate_fn,
        max_json_retries=max_json_retries,
        max_new_tokens=256,
    )


def generate_tourist_response_with_opening_check(
    messages: list[dict[str, Any]],
    category: str | None = None,
    current_place_name: str | None = None,
    db_path: str = DEFAULT_DB_PATH,
    current_time: datetime | None = None,
    generate_fn: Callable[[list[dict[str, Any]]], str] | None = None,
    max_retries: int = DEFAULT_MAX_CLOSED_RETRIES,
    max_json_retries: int = DEFAULT_MAX_JSON_RETRIES,
    origin_coords: tuple[float, float] | None = None,
) -> tuple[TouristResponse, bool, str, int]:
    """Útitervet generál nyitvatartás-ellenőrzéssel és távolság szerinti visszacsatolási hurokkal."""
    check_time = current_time or datetime.now()
    current_messages = list(messages)
    rejected_places: set[str] = set()

    for attempt in range(1, max_retries + 2):
        response = generate_structured_response(
            messages=current_messages,
            response_model=TouristResponse,
            generate_fn=generate_fn,
            max_json_retries=max_json_retries,
        )
        is_open, status_msg = is_place_open(
            response.ajanlott_kovetkezo_megallo,
            db_path=db_path,
            current_time=check_time,
        )
        if is_open:
            return response, True, status_msg, attempt

        if attempt > max_retries:
            return response, False, status_msg, attempt

        rejected_places.add(response.ajanlott_kovetkezo_megallo)

        matched_current = find_place(
            current_place_name or response.felismert_hely,
            db_path=db_path,
        )
        target_category = category
        if not target_category and matched_current:
            target_category = matched_current["category"]
        if not target_category:
            matched_suggested = find_place(
                response.ajanlott_kovetkezo_megallo,
                db_path=db_path,
            )
            if matched_suggested:
                target_category = matched_suggested["category"]

        if not target_category:
            return response, False, status_msg, attempt

        effective_origin = origin_coords
        if effective_origin is None and matched_current is not None:
            effective_origin = (matched_current["lat"], matched_current["lon"])

        base_excludes = [
            name
            for name in (current_place_name, response.felismert_hely)
            if name
        ]
        open_places = get_open_places_in_category(
            target_category,
            db_path=db_path,
            current_time=check_time,
            exclude_names=[*base_excludes, *rejected_places],
            origin_coords=effective_origin,
        )
        if not open_places:
            open_places = get_open_places_in_category(
                target_category,
                db_path=db_path,
                current_time=check_time,
                exclude_names=base_excludes,
                origin_coords=effective_origin,
            )
        if not open_places:
            return response, False, status_msg, attempt

        open_place_names = [place["name"] for place in open_places]
        feedback_msg = build_closed_place_feedback(
            response.ajanlott_kovetkezo_megallo,
            open_place_names,
        )
        current_messages.append(
            {"role": "assistant", "content": response.model_dump_json()}
        )
        current_messages.append({"role": "system", "content": feedback_msg})

    raise RuntimeError("Váratlan állapot a nyitvatartás-ellenőrzési hurokban.")


def _find_place_by_coordinates(
    coords: tuple[float, float],
    db_path: str = DEFAULT_DB_PATH,
    max_distance_km: float = 0.5,
) -> dict[str, Any] | None:
    """Megkeresi a koordinátákhoz legközelebbi adatbázis-helyszínt a megadott körzetben."""
    img_lat, img_lon = coords
    best_place: dict[str, Any] | None = None
    best_distance = float("inf")

    for place in get_all_places(db_path):
        distance = haversine_distance(img_lat, img_lon, place["lat"], place["lon"])
        if distance <= max_distance_km and distance < best_distance:
            best_distance = distance
            best_place = place

    return best_place


def run_tourist_pipeline(
    image_source: str | Path | Image.Image,
    audio_path: str | None = None,
    user_text: str | None = None,
    coords: tuple[float, float] | None = None,
    db_path: str = DEFAULT_DB_PATH,
    current_time: datetime | None = None,
    whisper_device: str | None = None,
    generate_fn: Callable[[list[dict[str, Any]]], str] | None = None,
    max_closed_retries: int = DEFAULT_MAX_CLOSED_RETRIES,
    max_json_retries: int = DEFAULT_MAX_JSON_RETRIES,
) -> PipelineResult:
    """Lefuttatja a teljes multimodális turisztikai asszisztens pipeline-t."""
    known_place_names: list[str] | None = None
    try:
        known_place_names = [place["name"] for place in get_all_places(db_path)]
    except FileNotFoundError:
        known_place_names = None

    image_check = check_image_landmark(
        image_source=image_source,
        generate_fn=generate_fn,
        max_json_retries=max_json_retries,
        known_places=known_place_names,
    )
    passed, guardrail_msg = evaluate_image_guardrail(image_check)
    if not passed:
        return PipelineResult(
            success=False,
            image_check=image_check,
            tourist_response=None,
            error_message=guardrail_msg,
            attempts=0,
        )

    transcript: dict[str, str] | None = None
    if audio_path:
        transcript = transcribe_audio(audio_path, device=whisper_device)

    effective_coords = coords
    if effective_coords is None:
        coord_input = (
            str(image_source) if isinstance(image_source, Path) else image_source
        )
        effective_coords = get_image_coordinates(coord_input)

    matched_place = (
        find_place(image_check.detected_name, db_path=db_path)
        if image_check.detected_name
        else None
    )
    if matched_place is None and effective_coords is not None:
        matched_place = _find_place_by_coordinates(
            effective_coords,
            db_path=db_path,
        )

    origin_coords: tuple[float, float] | None = None
    if effective_coords is not None:
        if matched_place is None or (
            haversine_distance(
                effective_coords[0],
                effective_coords[1],
                matched_place["lat"],
                matched_place["lon"],
            )
            <= 50.0
        ):
            origin_coords = effective_coords
    if origin_coords is None and matched_place is not None:
        origin_coords = (matched_place["lat"], matched_place["lon"])

    check_time = current_time or datetime.now()
    time_label = check_time.strftime("%H:%M")

    category: str | None = matched_place["category"] if matched_place else None
    place_name = (
        matched_place["name"]
        if matched_place
        else (image_check.detected_name or "Ismeretlen látványosság")
    )

    same_category_places: list[dict[str, Any]] = []
    candidate_lines: list[str] = []
    if category:
        if origin_coords is not None:
            same_category_places = get_places_by_category_sorted_by_distance(
                category,
                origin_coords[0],
                origin_coords[1],
                db_path=db_path,
                exclude_id=matched_place["id"] if matched_place else None,
            )
        else:
            same_category_places = get_places_by_category(
                category,
                db_path=db_path,
                exclude_id=matched_place["id"] if matched_place else None,
            )

        for idx, place in enumerate(same_category_places, start=1):
            if "distance_km" in place:
                candidate_lines.append(
                    f"{idx}. {place['name']} | távolság: {place['distance_km']:.2f} km "
                    f"(kb. {place['estimated_walk_minutes']} perc séta) | "
                    f"jegyár: {place['ticket_price_huf']} Ft | {place['description']}"
                )
            else:
                candidate_lines.append(
                    f"{idx}. {place['name']} | jegyár: {place['ticket_price_huf']} Ft | "
                    f"{place['description']}"
                )

    context_parts = [
        f"Aktuális időpont: {time_label}",
        f"Felismert helyszín: {place_name}",
    ]
    if matched_place:
        context_parts.append(
            f"Adatbázis adatok: kategória={matched_place['category']}, "
            f"jegyár={matched_place['ticket_price_huf']} Ft, "
            f"leírás={matched_place['description']}"
        )
    if transcript:
        context_parts.append(
            f"A turista hangüzenete ({transcript['language']}): '{transcript['text']}'"
        )
    if user_text:
        context_parts.append(f"A turista szöveges megjegyzése: '{user_text}'")
    if candidate_lines:
        context_parts.append(
            "Ugyanabba a kategóriába tartozó választható következő megállók távolság "
            "szerinti sorrendben (1. a legközelebbi):\n" + "\n".join(candidate_lines)
        )

    image_pil = _open_rgb_image(image_source)
    system_prompt = build_json_system_prompt(
        TouristResponse,
        "Te egy profi magyar idegenvezető asszisztens vagy. Készíts pontos, "
        "szórakoztató és megbízható választ a képen látható nevezetességről. "
        "A következő megálló kiválasztásánál MINDIG a távolság az elsődleges szempont: "
        "először a legközelebbi látványosságot válaszd a megadott listából! "
        "Ha a rendszer később jelzi, hogy a kiválasztott hely zárva van, "
        "akkor válaszd a megadott nyitva tartó lista legelső (legközelebbi) elemét!",
    )

    closest_instruction = ""
    if same_category_places and "distance_km" in same_category_places[0]:
        closest_place = same_category_places[0]
        closest_instruction = (
            f"\nA legközelebbi jelölt a listában az 1. helyen szereplő "
            f"'{closest_place['name']}' ({closest_place['distance_km']:.2f} km, "
            f"kb. {closest_place['estimated_walk_minutes']} perc séta) – elsőként ezt válaszd "
            f"az `ajanlott_kovetkezo_megallo` mezőbe, a `becsult_ido_percben` értéke pedig "
            f"legyen {closest_place['estimated_walk_minutes']}!"
        )

    user_prompt = (
        "\n\n".join(context_parts)
        + closest_instruction
        + "\n\nVálaszolj pontosan egy kitöltött JSON objektummal, amely tartalmazza a "
        "`felismert_hely`, `tortenelem_roviden`, `ajanlott_kovetkezo_megallo`, "
        "`becsult_ido_percben` és `megjegyzes` mezőket magyar nyelvű értékekkel!"
    )

    itinerary_messages: list[dict[str, Any]] = [
        {"role": "system", "content": system_prompt},
        {
            "role": "user",
            "content": [
                {
                    "type": "image",
                    "image": image_pil,
                    "min_pixels": MIN_PIXELS,
                    "max_pixels": MAX_PIXELS,
                },
                {"type": "text", "text": user_prompt},
            ],
        },
    ]

    tourist_response, is_open, status_msg, attempts = (
        generate_tourist_response_with_opening_check(
            messages=itinerary_messages,
            category=category,
            current_place_name=place_name,
            db_path=db_path,
            current_time=check_time,
            generate_fn=generate_fn,
            max_retries=max_closed_retries,
            max_json_retries=max_json_retries,
            origin_coords=origin_coords,
        )
    )

    next_stop_distance_km: float | None = None
    if origin_coords is not None:
        next_place = find_place(
            tourist_response.ajanlott_kovetkezo_megallo,
            db_path=db_path,
        )
        if next_place is not None:
            next_stop_distance_km = round(
                haversine_distance(
                    origin_coords[0],
                    origin_coords[1],
                    next_place["lat"],
                    next_place["lon"],
                ),
                3,
            )

    return PipelineResult(
        success=True,
        image_check=image_check,
        tourist_response=tourist_response,
        transcript=transcript,
        matched_place=matched_place,
        is_next_stop_open=is_open,
        next_stop_distance_km=next_stop_distance_km,
        opening_status_message=status_msg,
        attempts=attempts,
    )
