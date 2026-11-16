import json
import math
import re
import sqlite3
import warnings

from PIL import Image
from qwen_vl_utils import process_vision_info
import torch
from transformers import (
    AutoProcessor,
    BitsAndBytesConfig,
    Qwen2_5_VLForConditionalGeneration,
    logging as hf_logging,
)

from database import create_and_populate_db
from location import get_image_coordinates

warnings.filterwarnings("ignore", category=UserWarning)
hf_logging.set_verbosity_error()

DB_PATH = "tourism.db"
VISION_MODEL_ID = "Qwen/Qwen2.5-VL-3B-Instruct"
MIN_PIXELS = 256 * 28 * 28
MAX_PIXELS = 512 * 28 * 28

_vision_model = None
_vision_processor = None


def _load_vision_model():
    """Lazy-load Qwen2.5-VL-3B-Instruct model and processor."""
    global _vision_model, _vision_processor
    if _vision_model is not None and _vision_processor is not None:
        return _vision_model, _vision_processor

    has_cuda = torch.cuda.is_available()
    model_dtype = torch.float16 if has_cuda else torch.float32
    bnb_config = (
        BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
        )
        if has_cuda
        else None
    )

    _vision_model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        VISION_MODEL_ID,
        quantization_config=bnb_config,
        dtype=model_dtype,
        device_map="auto" if has_cuda else None,
        low_cpu_mem_usage=True,
    )
    _vision_processor = AutoProcessor.from_pretrained(
        VISION_MODEL_ID,
        min_pixels=MIN_PIXELS,
        max_pixels=MAX_PIXELS,
    )
    return _vision_model, _vision_processor


def get_places_from_db(db_path: str = DB_PATH) -> list[dict]:
    """Ensure the database is populated and fetch all places."""
    create_and_populate_db()
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM places")
    rows = [dict(row) for row in cursor.fetchall()]
    conn.close()
    return rows


def haversine_distance(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Calculate great-circle distance in kilometers between two GPS coordinates."""
    r = 6371.0
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lon2 - lon1)

    a = (
        math.sin(dphi / 2.0) ** 2
        + math.cos(phi1) * math.cos(phi2) * math.sin(dlambda / 2.0) ** 2
    )
    return 2.0 * r * math.atan2(math.sqrt(a), math.sqrt(1.0 - a))


def recognize_building(
    image_source: str | Image.Image,
    coords: tuple[float, float] | None = None,
    db_path: str = DB_PATH,
) -> str:
    """
    Recognize the building in the image using Qwen2.5-VL-3B-Instruct and the
    extracted GPS coordinates to find the exact match from the database.
    Returns the result in JSON format.
    """
    if coords is None:
        coords = get_image_coordinates(image_source)

    image_pil = (
        Image.open(image_source).convert("RGB")
        if isinstance(image_source, str)
        else image_source.convert("RGB")
    )

    places = get_places_from_db(db_path)

    if coords is not None:
        img_lat, img_lon = coords
        for place in places:
            place["distance_km"] = round(
                haversine_distance(img_lat, img_lon, place["lat"], place["lon"]), 4
            )
        places_sorted = sorted(places, key=lambda p: p["distance_km"])
        location_context = (
            f"A képből kinyert GPS koordináták: lat={img_lat:.6f}, lon={img_lon:.6f}"
        )
        candidates_text = "\n".join(
            [
                f"- id: {p['id']} | név: {p['name']} | kategória: {p['category']} | "
                f"lat: {p['lat']}, lon: {p['lon']} | távolság (km): {p['distance_km']}"
                for p in places_sorted
            ]
        )
    else:
        places_sorted = places
        location_context = "A képből kinyert GPS koordináták: Nem érhető el"
        candidates_text = "\n".join(
            [
                f"- id: {p['id']} | név: {p['name']} | kategória: {p['category']} | "
                f"lat: {p['lat']}, lon: {p['lon']}"
                for p in places_sorted
            ]
        )

    prompt = (
        "Nézd meg a képet, ismerd fel a rajta látható épületet vagy nevezetességet, "
        "és a képből kinyert GPS koordináták segítségével válaszd ki a pontos egyezést "
        "az alábbi adatbázis-helyszínek közül!\n\n"
        f"{location_context}\n\n"
        f"Adatbázisban szereplő helyszínek:\n{candidates_text}\n\n"
        "Válaszként CSAK egy érvényes JSON objektumot adj vissza a felismert helyszínnel:\n"
        '{"id": "<id>", "name": "<name>"}'
    )

    model, processor = _load_vision_model()
    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "image",
                    "image": image_pil,
                    "min_pixels": MIN_PIXELS,
                    "max_pixels": MAX_PIXELS,
                },
                {"type": "text", "text": prompt},
            ],
        }
    ]

    text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    image_inputs, video_inputs = process_vision_info(messages)

    target_device = "cuda" if torch.cuda.is_available() else "cpu"
    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    ).to(target_device)

    with torch.inference_mode():
        generated_ids = model.generate(**inputs, max_new_tokens=80, do_sample=False)

    generated_ids_trimmed = [
        out_ids[len(in_ids) :]
        for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]
    raw_output = processor.batch_decode(
        generated_ids_trimmed,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )[0].strip()

    matched_place = None
    # Try parsing JSON from model output first
    json_match = re.search(r"\{.*?\}", raw_output, re.DOTALL)
    if json_match:
        try:
            parsed = json.loads(json_match.group(0))
            parsed_id = str(parsed.get("id", "")).lower()
            parsed_name = str(parsed.get("name", "")).lower()
            for p in places_sorted:
                if p["id"].lower() == parsed_id or p["name"].lower() == parsed_name:
                    matched_place = p
                    break
        except json.JSONDecodeError:
            pass

    # Fallback: match by id or name substring in the model response
    if matched_place is None:
        raw_lower = raw_output.lower()
        for p in places_sorted:
            if p["id"].lower() in raw_lower or p["name"].lower() in raw_lower:
                matched_place = p
                break

    # Fallback to closest coordinate match if within 0.5 km
    if matched_place is None and coords is not None and places_sorted:
        if places_sorted[0].get("distance_km", float("inf")) <= 0.5:
            matched_place = places_sorted[0]

    result = {
        "image_coordinates": (
            {"lat": coords[0], "lon": coords[1]} if coords is not None else None
        ),
        "match": matched_place,
    }
    return json.dumps(result, ensure_ascii=False, indent=2)
