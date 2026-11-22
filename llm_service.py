""""Legkisebb memóriaterhelésű lokális (CLIP + Qwen2.5-0.5B) LLM és pipeline szolgáltatás."""

from __future__ import annotations

from collections.abc import Callable
from datetime import datetime
import gc
import json
import logging
from pathlib import Path
import re
from typing import Any, TypeVar
import warnings

from PIL import Image
from pydantic import BaseModel, ValidationError

from audio_processor import clear_whisper_cache, transcribe_audio
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

CLIP_MODEL_ID: str = "openai/clip-vit-base-patch32"
TEXT_MODEL_ID: str = "Qwen/Qwen2.5-0.5B-Instruct"

LANDMARK_CONFIDENCE_THRESHOLD: float = 0.25
DEFAULT_MAX_JSON_RETRIES: int = 1
DEFAULT_MAX_CLOSED_RETRIES: int = 1

T = TypeVar("T", bound=BaseModel)

_CODE_FENCE_PATTERN = re.compile(
    r"```(?:json)?\s*(.*?)\s*```",
    flags=re.DOTALL | re.IGNORECASE,
)

_PLACE_CLIP_HINTS: dict[str, str] = {
    "országház": "Hungarian Parliament Building by the Danube in Budapest",
    "parlament": "Hungarian Parliament Building by the Danube in Budapest",
    "lánchíd": "Szechenyi Chain Bridge suspension bridge over the Danube in Budapest",
    "széchenyi lánchíd": "Szechenyi Chain Bridge suspension bridge over the Danube in Budapest",
    "halászbástya": "Fisherman's Bastion neo-Gothic terraces in Budapest",
    "budai vár": "Buda Castle royal palace in Budapest",
    "mátyás templom": "Matthias Church gothic church in Budapest",
    "hősök tere": "Heroes' Square memorial columns in Budapest",
    "szent istván-bazilika": "St. Stephen's Basilica dome in Budapest",
    "szent istván bazilika": "St. Stephen's Basilica dome in Budapest",
    "citadella": "Citadella fortress on Gellert Hill in Budapest",
    "margitsziget": "Margaret Island park and water tower in Budapest",
}

_clip_model: Any = None
_clip_processor: Any = None
_loaded_clip_model_id: str | None = None

_text_model: Any = None
_text_tokenizer: Any = None
_loaded_text_model_id: str | None = None

_default_text_service: LocalTextLLMService | None = None


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


def _empty_torch_cache_and_gc() -> None:
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except ImportError:  # pragma: no cover
        pass
    gc.collect()


def clear_clip_cache() -> None:
    """Felszabadítja a betöltött CLIP modellt és a gyorsítótárakat."""
    global _clip_model, _clip_processor, _loaded_clip_model_id
    _clip_model = None
    _clip_processor = None
    _loaded_clip_model_id = None
    _empty_torch_cache_and_gc()


def clear_qwen_cache() -> None:
    """Kompatibilitási név: üríti a gyorsítótárakat és futtatja a garbage collectort."""
    clear_clip_cache()


def clear_text_llm_cache() -> None:
    """Felszabadítja a betöltött szöveges LLM modellt és a gyorsítótárakat."""
    global _text_model, _text_tokenizer, _loaded_text_model_id
    _text_model = None
    _text_tokenizer = None
    _loaded_text_model_id = None
    _empty_torch_cache_and_gc()


def load_clip_model(
    model_id: str = CLIP_MODEL_ID,
) -> tuple[Any, Any]:
    """Lustán betölti a kis CLIP modellt a látványosság-ellenőrzéshez."""
    global _clip_model, _clip_processor, _loaded_clip_model_id

    if (
        _clip_model is not None
        and _clip_processor is not None
        and _loaded_clip_model_id == model_id
    ):
        return _clip_model, _clip_processor

    if _clip_model is not None:
        clear_clip_cache()

    logging.getLogger("torch.utils.flop_counter").setLevel(logging.ERROR)
    import torch
    from transformers import CLIPModel, CLIPProcessor, logging as hf_logging

    warnings.filterwarnings("ignore", category=UserWarning)
    hf_logging.set_verbosity_error()

    has_cuda = torch.cuda.is_available()
    model_dtype = torch.float16 if has_cuda else torch.float32

    _clip_processor = CLIPProcessor.from_pretrained(model_id)
    _clip_model = CLIPModel.from_pretrained(
        model_id,
        dtype=model_dtype,
        device_map="auto" if has_cuda else None,
        low_cpu_mem_usage=True,
    )
    _clip_model.eval()
    _loaded_clip_model_id = model_id
    return _clip_model, _clip_processor


def load_text_llm_model(
    model_id: str = TEXT_MODEL_ID,
) -> tuple[Any, Any]:
    """Lustán betölti a legkisebb szöveges Qwen2.5-0.5B modellt."""
    global _text_model, _text_tokenizer, _loaded_text_model_id

    if (
        _text_model is not None
        and _text_tokenizer is not None
        and _loaded_text_model_id == model_id
    ):
        return _text_model, _text_tokenizer

    if _text_model is not None:
        clear_text_llm_cache()

    logging.getLogger("torch.utils.flop_counter").setLevel(logging.ERROR)
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, logging as hf_logging

    warnings.filterwarnings("ignore", category=UserWarning)
    hf_logging.set_verbosity_error()

    has_cuda = torch.cuda.is_available()
    model_dtype = torch.float16 if has_cuda else torch.float32

    _text_tokenizer = AutoTokenizer.from_pretrained(model_id)
    _text_model = AutoModelForCausalLM.from_pretrained(
        model_id,
        dtype=model_dtype,
        device_map="auto" if has_cuda else None,
        low_cpu_mem_usage=True,
    )
    _text_model.eval()
    _loaded_text_model_id = model_id
    return _text_model, _text_tokenizer


class LocalTextLLMService:
    """Könnyű lokális szöveges LLM szolgáltatás JSON/útiterv generáláshoz."""

    def __init__(self, model_id: str = TEXT_MODEL_ID) -> None:
        self.model_id = model_id

    def generate(
        self,
        messages: list[dict[str, Any]],
        max_new_tokens: int = 160,
    ) -> str:
        """Szöveges választ generál kis tokenlimittel és alacsony RAM-igénnyel."""
        import torch

        model, tokenizer = load_text_llm_model(model_id=self.model_id)

        normalized_messages: list[dict[str, str]] = []
        for msg in messages:
            role = str(msg.get("role", "user"))
            content = msg.get("content", "")
            if isinstance(content, list):
                text_parts = [
                    str(item.get("text", ""))
                    for item in content
                    if isinstance(item, dict) and item.get("type") == "text"
                ]
                joined_text = "\n".join(part for part in text_parts if part).strip()
                normalized_messages.append({"role": role, "content": joined_text})
            else:
                normalized_messages.append({"role": role, "content": str(content)})

        prompt_text = tokenizer.apply_chat_template(
            normalized_messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        target_device = "cuda" if torch.cuda.is_available() else "cpu"
        model_inputs = tokenizer([prompt_text], return_tensors="pt").to(target_device)

        with torch.inference_mode():
            generated_ids = model.generate(
                **model_inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
            )

        trimmed_ids = [
            out_ids[len(in_ids) :]
            for in_ids, out_ids in zip(model_inputs.input_ids, generated_ids)
        ]
        return tokenizer.batch_decode(trimmed_ids, skip_special_tokens=True)[0].strip()


def get_default_text_llm_service() -> LocalTextLLMService:
    """Visszaadja az alapértelmezett singleton LocalTextLLMService példányt."""
    global _default_text_service
    if _default_text_service is None:
        _default_text_service = LocalTextLLMService()
    return _default_text_service


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
    max_new_tokens: int = 160,
) -> T:
    """Meghívja a kis szöveges modellt és validálja a JSON választ."""
    current_messages = list(messages)
    caller = generate_fn or (
        lambda msgs: get_default_text_llm_service().generate(
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
                current_messages.append({"role": "assistant", "content": raw_output})
                current_messages.append(
                    {
                        "role": "user",
                        "content": (
                            f"Érvénytelen JSON ({exc}). Csak egy kitöltött JSON objektumot adj vissza "
                            "minden egyéb szöveg nélkül:\n"
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


def _place_to_clip_prompt(place_name: str) -> str:
    """Angol + budapesti kontextust ad a CLIP promptnak a pontosabb felismeréshez."""
    cleaned = " ".join(place_name.lower().strip().split())
    if cleaned in _PLACE_CLIP_HINTS:
        return f"a clear photograph of {_PLACE_CLIP_HINTS[cleaned]}"
    for key, hint in _PLACE_CLIP_HINTS.items():
        if key in cleaned:
            return f"a clear photograph of {hint}"
    return f"a clear photograph of {place_name}, a landmark in Hungary"


def check_image_landmark(
    image_source: str | Path | Image.Image,
    generate_fn: Callable[[list[dict[str, Any]]], str] | None = None,
    max_json_retries: int = DEFAULT_MAX_JSON_RETRIES,
    known_places: list[str] | None = None,
) -> ImageCheckResult:
    """Könnyű CLIP-alapú látványosság-ellenőrzés a 3B-s VL modell kihagyásával."""
    if generate_fn is not None:
        image_pil = _open_rgb_image(image_source)
        system_prompt = build_json_system_prompt(
            ImageCheckResult,
            "Döntsd el, hogy a képen turisztikai látványosság látható-e.",
        )
        messages: list[dict[str, Any]] = [
            {"role": "system", "content": system_prompt},
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image_pil},
                    {"type": "text", "text": "Látványosság látható a képen?"},
                ],
            },
        ]
        return generate_structured_response(
            messages=messages,
            response_model=ImageCheckResult,
            generate_fn=generate_fn,
            max_json_retries=max_json_retries,
            max_new_tokens=96,
        )

    import torch

    image_pil = _open_rgb_image(image_source).resize((224, 224))
    model, processor = load_clip_model(CLIP_MODEL_ID)

    place_candidates = [p.strip() for p in (known_places or []) if p and p.strip()]
    non_landmark_labels = [
        "a blurry photo of everyday clutter, a blank screen, or a random close-up object",
        "not a tourist landmark or building",
    ]
    landmark_prompts = [_place_to_clip_prompt(name) for name in place_candidates] + [
        *non_landmark_labels,
    ]
    non_landmark_start_idx = len(place_candidates)

    target_device = "cuda" if torch.cuda.is_available() else "cpu"
    inputs = processor(
        text=landmark_prompts,
        images=image_pil,
        return_tensors="pt",
        padding=True,
    ).to(target_device)

    with torch.inference_mode():
        outputs = model(**inputs)
        probs = outputs.logits_per_image.softmax(dim=1)[0].detach().cpu().tolist()

    best_idx = max(range(len(probs)), key=lambda i: probs[i])
    best_prob = float(probs[best_idx])
    is_non_landmark = best_idx >= non_landmark_start_idx

    detected_name: str | None = None
    if not is_non_landmark and best_idx < len(place_candidates):
        detected_name = place_candidates[best_idx]

    is_landmark = (
        (not is_non_landmark)
        and (detected_name is not None)
        and (best_prob >= LANDMARK_CONFIDENCE_THRESHOLD)
    )
    reason = (
        f"CLIP legjobb találat: {detected_name or 'ismeretlen látványosság'} ({best_prob:.0%})."
        if is_landmark
        else f"CLIP szerint nem egyértelmű látványosság ({best_prob:.0%})."
    )

    return ImageCheckResult(
        is_landmark=is_landmark,
        confidence=round(best_prob, 4),
        detected_name=detected_name,
        reason=reason,
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
    """Útitervet generál nyitvatartás-ellenőrzéssel és visszacsatolási hurokkal."""
    check_time = current_time or datetime.now()
    current_messages = list(messages)
    rejected_places: set[str] = set()

    for attempt in range(1, max_retries + 2):
        response = generate_structured_response(
            messages=current_messages,
            response_model=TouristResponse,
            generate_fn=generate_fn,
            max_json_retries=max_json_retries,
            max_new_tokens=160,
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
    if generate_fn is None:
        clear_clip_cache()

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
        clear_whisper_cache()
        gc.collect()

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

        for idx, place in enumerate(same_category_places[:3], start=1):
            if "distance_km" in place:
                candidate_lines.append(
                    f"{idx}. {place['name']} | {place['distance_km']:.2f} km "
                    f"({place['estimated_walk_minutes']} perc)"
                )
            else:
                candidate_lines.append(f"{idx}. {place['name']}")

    context_parts = [
        f"Időpont: {time_label}",
        f"Helyszín: {place_name}",
    ]
    if matched_place:
        context_parts.append(
            f"Kategória: {matched_place['category']}, jegyár: {matched_place['ticket_price_huf']} Ft"
        )
    if transcript and transcript.get("text"):
        context_parts.append(f"Kérdés: '{transcript['text']}'")
    if user_text:
        context_parts.append(f"Megjegyzés: '{user_text}'")
    if candidate_lines:
        context_parts.append("Következő jelöltek:\n" + "\n".join(candidate_lines))

    system_prompt = build_json_system_prompt(
        TouristResponse,
        "Magyar idegenvezető asszisztens vagy. Válaszolj tömören, pontos JSON-nal. "
        "Mindig a legközelebbi megállót válaszd a jelöltek közül.",
    )

    closest_instruction = ""
    if same_category_places and "distance_km" in same_category_places[0]:
        closest_place = same_category_places[0]
        closest_instruction = (
            f"\nLegközelebbi megálló: '{closest_place['name']}', "
            f"becsult_ido_percben: {closest_place['estimated_walk_minutes']}."
        )

    user_prompt = (
        "\n".join(context_parts)
        + closest_instruction
        + "\nAdd meg a JSON mezőket magyarul."
    )

    itinerary_messages: list[dict[str, Any]] = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
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