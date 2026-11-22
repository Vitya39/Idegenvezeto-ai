from __future__ import annotations

from datetime import datetime
import gc
from io import BytesIO
import logging
from pathlib import Path
import tempfile

import folium
from gtts import gTTS
from PIL import Image
import streamlit as st
import streamlit.components.v1 as components

logging.getLogger("torch.utils.flop_counter").setLevel(logging.ERROR)

from audio_processor import (
    AudioProcessingError,
    DEFAULT_WHISPER_MODEL,
    clear_whisper_cache,
    transcribe_audio,
)
from database import create_and_populate_db
from llm_service import (
    StructuredOutputError,
    check_image_landmark,
    clear_clip_cache,
    clear_text_llm_cache,
    evaluate_image_guardrail,
    generate_tourist_response_with_opening_check,
)
from location import get_image_coordinates
from schemas import ImageCheckResult, PipelineResult, TouristResponse
from validator import (
    DEFAULT_DB_PATH,
    find_place,
    get_all_places,
    get_places_by_category,
    get_places_by_category_sorted_by_distance,
    haversine_distance,
    is_place_open,
)

DEFAULT_FALLBACK_COORDS: tuple[float, float] = (47.4979, 19.0402)

st.set_page_config(
    page_title="Multimodális Turisztikai Asszisztens",
    page_icon="📍",
    layout="wide",
)


@st.cache_resource
def init_db_once(db_path: str = DEFAULT_DB_PATH) -> None:
    """Biztosítja, hogy a tourism.db létezzen és tartalmazza a built_year mezőt."""
    create_and_populate_db(db_path=db_path)


def _cleanup_all_caches() -> None:
    """Minden modell-gyorsítótárat ürít és futtatja a garbage collectort."""
    clear_clip_cache()
    clear_whisper_cache()
    clear_text_llm_cache()
    gc.collect()


def _init_session_state() -> None:
    """Inicializálja a lépések közötti állapotváltozókat."""
    defaults = {
        "image_check": None,
        "coords": None,
        "used_fallback_coords": False,
        "matched_place": None,
        "origin_coords": None,
        "image_step_error": None,
        "transcript": None,
        "audio_direct_answer": None,
        "audio_step_error": None,
        "pipeline_result": None,
        "itinerary_step_error": None,
    }
    for key, val in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = val


def _reset_pipeline_state() -> None:
    """Törli a korábbi lépések eredményeit és üríti a gyorsítótárakat."""
    st.session_state.image_check = None
    st.session_state.coords = None
    st.session_state.used_fallback_coords = False
    st.session_state.matched_place = None
    st.session_state.origin_coords = None
    st.session_state.image_step_error = None
    st.session_state.transcript = None
    st.session_state.audio_direct_answer = None
    st.session_state.audio_step_error = None
    st.session_state.pipeline_result = None
    st.session_state.itinerary_step_error = None
    _cleanup_all_caches()


def format_built_year(year: int | None) -> str:
    """Formázza az építési évet (negatív érték esetén i. e.)."""
    if year is None:
        return "ismeretlen"
    if year < 0:
        return f"i. e. {abs(year)}"
    return str(year)


def synthesize_tts_mp3(text: str, lang: str = "hu") -> bytes | None:
    """Szövegből MP3 bájtfolyamot készít gTTS segítségével."""
    cleaned = " ".join(text.split()).strip()
    if not cleaned:
        return None
    tts = gTTS(text=cleaned, lang=lang)
    buf = BytesIO()
    tts.write_to_fp(buf)
    buf.seek(0)
    return buf.read()


def _build_direct_question_answer(
    question_text: str,
    place_name: str,
    matched_place: dict | None,
    tr: TouristResponse | None = None,
) -> str:
    """Célzott választ állít össze a hangban feltett kérdésre a helyszín adatai alapján."""
    q_low = question_text.lower()
    built_str = (
        format_built_year(matched_place.get("built_year"))
        if matched_place
        else "ismeretlen"
    )
    desc = (matched_place.get("description") or "").strip() if matched_place else ""
    opening = (matched_place.get("opening_hours") or "").strip() if matched_place else ""
    price = matched_place.get("ticket_price_huf") if matched_place else None

    asks_year = any(
        k in q_low
        for k in (
            "mikor épült",
            "melyik év",
            "évben épült",
            "hány éves",
            "mikori",
            "évszám",
            "épült",
            " év ",
            "év?",
        )
    ) or q_low.startswith("év")
    asks_country = any(
        k in q_low
        for k in (
            "melyik ország",
            "hol található",
            "melyik város",
            "ország",
            "város",
            "hol van",
        )
    )
    asks_hours = any(
        k in q_low for k in ("nyitva", "nyitvatartás", "mikor nyit", "mikor zár", "meddig")
    )
    asks_price = any(
        k in q_low for k in ("mennyibe", "belépő", "jegy", "ár", "ft", "forint", "kerül")
    )

    parts: list[str] = []
    if asks_year:
        parts.append(f"A(z) **{place_name}** építési éve: **{built_str}**.")
    if asks_country:
        if (
            "párizs" in desc.lower()
            or "eiffel" in place_name.lower()
            or "louvre" in place_name.lower()
        ):
            parts.append(f"A(z) **{place_name}** **Franciaországban** (Párizsban) található.")
        else:
            parts.append(f"A(z) **{place_name}** **Magyarországon** (Budapesten) található.")
    if asks_hours and opening:
        parts.append(f"Nyitvatartása: **{opening}**.")
    if asks_price and price is not None:
        parts.append(f"Belépőjegye: **{price} Ft**.")

    if not parts:
        summary = desc or (tr.tortenelem_roviden if tr else "")
        base = f"A(z) **{place_name}** (épült: **{built_str}**)"
        if summary:
            parts.append(f"{base}: {summary}")
        else:
            parts.append(f"{base}.")
        if tr and tr.megjegyzes:
            parts.append(tr.megjegyzes)
    elif tr and tr.megjegyzes:
        parts.append(tr.megjegyzes)

    return " ".join(parts).strip()


def _refresh_audio_direct_answer() -> None:
    """Frissíti a 2. lépésben megjelenő azonnali kérdés-választ a meglévő kép + hang alapján."""
    transcript = st.session_state.transcript
    if not transcript or not transcript.get("text"):
        st.session_state.audio_direct_answer = None
        return

    question_text = transcript["text"].strip()
    if not question_text:
        st.session_state.audio_direct_answer = None
        return

    matched_place = st.session_state.matched_place
    image_check: ImageCheckResult | None = st.session_state.image_check

    if matched_place is None and image_check is None:
        st.session_state.audio_direct_answer = (
            "Futtasd le az **1. Kép ellenőrzése** lépést is, hogy tudjam, melyik "
            "helyszínre vonatkozik a kérdésed!"
        )
        return

    place_name = (
        matched_place["name"]
        if matched_place
        else (
            image_check.detected_name
            if image_check and image_check.detected_name
            else "Ismeretlen helyszín"
        )
    )
    st.session_state.audio_direct_answer = _build_direct_question_answer(
        question_text=question_text,
        place_name=place_name,
        matched_place=matched_place,
        tr=None,
    )


def build_unified_response_text(result: PipelineResult, db_path: str = DEFAULT_DB_PATH) -> str:
    """Egységes választ készít: közvetlen válasz a kérdésre + következő célpont ajánlása."""
    tr = result.tourist_response
    if tr is None:
        return result.error_message or "Nem sikerült választ generálni."

    question_text = ""
    if result.transcript and result.transcript.get("text"):
        question_text = result.transcript["text"].strip()

    matched = result.matched_place or find_place(tr.felismert_hely, db_path=db_path)
    next_place = find_place(tr.ajanlott_kovetkezo_megallo, db_path=db_path)

    next_built_str = (
        format_built_year(next_place.get("built_year"))
        if next_place
        else "ismeretlen"
    )

    if question_text:
        direct_ans = _build_direct_question_answer(
            question_text=question_text,
            place_name=tr.felismert_hely,
            matched_place=matched,
            tr=tr,
        )
        answer_section = (
            f"**Kérdésed (*„{question_text}”*) – Válasz:** {direct_ans}\n\n"
            f"*Felismert helyszín:* **{tr.felismert_hely}** — {tr.tortenelem_roviden}"
        )
    else:
        built_str = (
            format_built_year(matched.get("built_year"))
            if matched
            else "ismeretlen"
        )
        answer_section = (
            f"**Helyszínismertető:** A fotón a(z) **{tr.felismert_hely}** látható "
            f"(épült: {built_str}). {tr.tortenelem_roviden}"
        )
        if tr.megjegyzes:
            answer_section += f" {tr.megjegyzes}"

    dist_info = (
        f" ({result.next_stop_distance_km:.2f} km)"
        if result.next_stop_distance_km is not None
        else ""
    )
    walk_info = (
        f", kb. {tr.becsult_ido_percben} perc séta"
        if tr.becsult_ido_percben
        else ""
    )
    open_info = (
        "jelenleg nyitva tart"
        if result.is_next_stop_open
        else "jelenleg zárva van vagy nyitvatartása ellenőrzendő"
    )

    recommendation_section = (
        f"**Következő ajánlott célpont:** Érdemes továbbmenned a(z) "
        f"**{tr.ajanlott_kovetkezo_megallo}** felé (épült: {next_built_str})"
        f"{dist_info}{walk_info}, amely {open_info}."
    )
    if next_place:
        if next_place.get("description"):
            recommendation_section += f" {next_place['description']}"
        if next_place.get("opening_hours") and next_place.get("ticket_price_huf") is not None:
            recommendation_section += (
                f" (Nyitvatartás: {next_place['opening_hours']}, "
                f"belépő: {next_place['ticket_price_huf']} Ft.)"
            )

    return f"{answer_section}\n\n{recommendation_section}"


def build_folium_map(
    origin_coords: tuple[float, float] | None,
    matched_place: dict | None,
    next_place: dict | None,
) -> folium.Map:
    """Elkészíti a Folium térképet a kiindulási ponttal, felismert és ajánlott hellyel."""
    points: list[tuple[float, float]] = []
    if origin_coords is not None:
        points.append(origin_coords)
    if matched_place is not None:
        points.append((matched_place["lat"], matched_place["lon"]))
    if next_place is not None:
        points.append((next_place["lat"], next_place["lon"]))

    center = points[0] if points else DEFAULT_FALLBACK_COORDS
    m = folium.Map(location=center, zoom_start=14, tiles="OpenStreetMap")

    if origin_coords is not None:
        folium.Marker(
            location=origin_coords,
            popup="<b>Fotó / kiindulópont</b>",
            tooltip="Kiindulási pont",
            icon=folium.Icon(color="blue", icon="camera", prefix="fa"),
        ).add_to(m)

    if matched_place is not None:
        m_coords = (matched_place["lat"], matched_place["lon"])
        built_txt = format_built_year(matched_place.get("built_year"))
        dist_txt = ""
        if origin_coords is not None:
            d_km = haversine_distance(
                origin_coords[0], origin_coords[1], m_coords[0], m_coords[1]
            )
            dist_txt = f"<br>Távolság: {d_km:.2f} km"
        folium.Marker(
            location=m_coords,
            popup=(
                f"<b>Felismert helyszín: {matched_place['name']}</b><br>"
                f"Épült: {built_txt}<br>"
                f"Nyitvatartás: {matched_place['opening_hours']}<br>"
                f"Belépő: {matched_place['ticket_price_huf']} Ft{dist_txt}"
            ),
            tooltip=f"Felismert: {matched_place['name']}",
            icon=folium.Icon(color="green", icon="landmark", prefix="fa"),
        ).add_to(m)

    if next_place is not None:
        n_coords = (next_place["lat"], next_place["lon"])
        built_txt = format_built_year(next_place.get("built_year"))
        dist_txt = ""
        if origin_coords is not None:
            d_km = haversine_distance(
                origin_coords[0], origin_coords[1], n_coords[0], n_coords[1]
            )
            dist_txt = f"<br>Távolság: {d_km:.2f} km"
        folium.Marker(
            location=n_coords,
            popup=(
                f"<b>Következő ajánlott célpont: {next_place['name']}</b><br>"
                f"Épült: {built_txt}<br>"
                f"Nyitvatartás: {next_place['opening_hours']}<br>"
                f"Belépő: {next_place['ticket_price_huf']} Ft{dist_txt}"
            ),
            tooltip=f"Ajánlott: {next_place['name']}",
            icon=folium.Icon(color="red", icon="flag", prefix="fa"),
        ).add_to(m)

    line_coords: list[tuple[float, float]] = []
    if origin_coords is not None:
        line_coords.append(origin_coords)
    elif matched_place is not None:
        line_coords.append((matched_place["lat"], matched_place["lon"]))

    if next_place is not None:
        line_coords.append((next_place["lat"], next_place["lon"]))

    if len(line_coords) == 2:
        folium.PolyLine(
            locations=line_coords,
            color="#d9381e",
            weight=4,
            opacity=0.85,
            dash_array="8",
        ).add_to(m)

    if len(points) >= 2:
        m.fit_bounds(points, padding=(25, 25))

    return m


def _save_uploaded_to_temp(uploaded_file, suffix: str) -> str:
    """Feltöltött Streamlit fájl mentése ideiglenes fájlba."""
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(uploaded_file.getvalue())
        return tmp.name


def _find_place_by_coordinates(
    coords: tuple[float, float],
    db_path: str = DEFAULT_DB_PATH,
    max_distance_km: float = 0.5,
) -> dict | None:
    """Megkeresi a koordinátákhoz legközelebbi adatbázis-helyszínt a megadott körzetben."""
    img_lat, img_lon = coords
    best_place: dict | None = None
    best_distance = float("inf")

    for place in get_all_places(db_path):
        distance = haversine_distance(img_lat, img_lon, place["lat"], place["lon"])
        if distance <= max_distance_km and distance < best_distance:
            best_distance = distance
            best_place = place

    return best_place


def _collect_db_candidates(
    category: str | None,
    origin_coords: tuple[float, float] | None,
    exclude_id: int | str | None,
    exclude_name: str | None,
    db_path: str = DEFAULT_DB_PATH,
) -> list[dict]:
    """Összegyűjti a létező adatbázis-helyszíneket a következő megálló kiválasztásához."""
    candidates: list[dict] = []

    if category:
        if origin_coords is not None:
            candidates = get_places_by_category_sorted_by_distance(
                category,
                origin_coords[0],
                origin_coords[1],
                db_path=db_path,
                exclude_id=exclude_id,
            )
        else:
            candidates = get_places_by_category(
                category,
                db_path=db_path,
                exclude_id=exclude_id,
            )

    if not candidates:
        all_places = get_all_places(db_path)
        filtered: list[dict] = []
        for p in all_places:
            if exclude_id is not None and p.get("id") == exclude_id:
                continue
            if exclude_name and p.get("name", "").strip().lower() == exclude_name.strip().lower():
                continue
            item = dict(p)
            if origin_coords is not None:
                dist_km = haversine_distance(
                    origin_coords[0], origin_coords[1], item["lat"], item["lon"]
                )
                item["distance_km"] = round(dist_km, 3)
                item["estimated_walk_minutes"] = max(1, int(round((dist_km / 4.5) * 60)))
            filtered.append(item)

        if origin_coords is not None:
            filtered.sort(key=lambda x: x.get("distance_km", float("inf")))
        candidates = filtered

    return candidates


def _pick_next_db_place(
    suggested_name: str | None,
    candidates: list[dict],
    current_place_name: str | None,
    db_path: str = DEFAULT_DB_PATH,
) -> dict | None:
    """Biztosítja, hogy a következő megálló mindig létező adatbázis-rekord legyen."""
    norm_current = (current_place_name or "").strip().lower()

    if suggested_name:
        db_match = find_place(suggested_name, db_path=db_path)
        if db_match and db_match.get("name", "").strip().lower() != norm_current:
            for c in candidates:
                if c.get("id") == db_match.get("id") or (
                    c.get("name", "").strip().lower() == db_match.get("name", "").strip().lower()
                ):
                    merged = dict(db_match)
                    for k in ("distance_km", "estimated_walk_minutes"):
                        if k in c:
                            merged[k] = c[k]
                    return merged
            return db_match

    for c in candidates:
        if c.get("name", "").strip().lower() != norm_current:
            return c

    for p in get_all_places(db_path):
        if p.get("name", "").strip().lower() != norm_current:
            return p

    return None


def _build_fallback_tourist_response(
    place_name: str,
    matched_place: dict | None,
    chosen_next: dict | None,
    transcript: dict | None,
) -> TouristResponse:
    """Biztonságos adatbázis-alapú TouristResponse, ha a 0.5B modell JSON-ja hibás."""
    history_text = (
        (matched_place.get("description") or "").strip()
        if matched_place
        else f"A(z) {place_name} ismert látványosság."
    )
    if not history_text:
        history_text = f"A(z) {place_name} ismert látványosság."

    next_name = (
        chosen_next["name"]
        if chosen_next and chosen_next.get("name")
        else place_name
    )
    walk_mins = (
        int(chosen_next["estimated_walk_minutes"])
        if chosen_next and chosen_next.get("estimated_walk_minutes")
        else 10
    )

    q_text = (transcript.get("text") or "").strip() if transcript else ""
    note = (
        _build_direct_question_answer(q_text, place_name, matched_place, tr=None)
        if q_text
        else "Adatbázisból összeállított ajánlás."
    )

    return TouristResponse(
        felismert_hely=place_name,
        tortenelem_roviden=history_text,
        ajanlott_kovetkezo_megallo=next_name,
        becsult_ido_percben=max(1, walk_mins),
        megjegyzes=note,
    )


def _run_image_step(uploaded_image, db_path: str = DEFAULT_DB_PATH) -> None:
    """1. lépés: Opcionális EXIF GPS + CLIP képellenőrzés (GPS nélkül is működik)."""
    st.session_state.image_step_error = None
    st.session_state.pipeline_result = None
    st.session_state.itinerary_step_error = None
    st.session_state.used_fallback_coords = False

    img_suffix = Path(uploaded_image.name).suffix.lower() or ".jpg"
    tmp_img_path = _save_uploaded_to_temp(uploaded_image, img_suffix)

    try:
        exif_coords = get_image_coordinates(tmp_img_path)

        known_place_names: list[str] | None = None
        try:
            known_place_names = [place["name"] for place in get_all_places(db_path)]
        except FileNotFoundError:
            known_place_names = None

        with Image.open(tmp_img_path) as img:
            pil_img = img.convert("RGB")

        image_check = check_image_landmark(
            image_source=pil_img,
            known_places=known_place_names,
        )
        clear_clip_cache()
        gc.collect()

        passed, guardrail_msg = evaluate_image_guardrail(image_check)
        if not passed:
            st.session_state.image_step_error = guardrail_msg
            return

        matched_place = (
            find_place(image_check.detected_name, db_path=db_path)
            if image_check.detected_name
            else None
        )
        if matched_place is None and exif_coords is not None:
            matched_place = _find_place_by_coordinates(exif_coords, db_path=db_path)

        if exif_coords is not None:
            effective_coords = exif_coords
            if matched_place is None or (
                haversine_distance(
                    exif_coords[0],
                    exif_coords[1],
                    matched_place["lat"],
                    matched_place["lon"],
                )
                <= 50.0
            ):
                origin_coords = exif_coords
            else:
                origin_coords = (matched_place["lat"], matched_place["lon"])
        elif matched_place is not None:
            effective_coords = (matched_place["lat"], matched_place["lon"])
            origin_coords = effective_coords
            st.session_state.used_fallback_coords = True
        else:
            effective_coords = DEFAULT_FALLBACK_COORDS
            origin_coords = DEFAULT_FALLBACK_COORDS
            st.session_state.used_fallback_coords = True

        st.session_state.coords = effective_coords
        st.session_state.image_check = image_check
        st.session_state.matched_place = matched_place
        st.session_state.origin_coords = origin_coords
        _refresh_audio_direct_answer()
    finally:
        try:
            Path(tmp_img_path).unlink(missing_ok=True)
        except OSError:
            pass
        clear_clip_cache()
        gc.collect()


def _run_audio_step(active_audio, whisper_dev: str | None) -> None:
    """2. lépés: Hangátírás Whisperrel + azonnali válasz a kérdésre, majd gyorsítótár ürítése."""
    st.session_state.audio_step_error = None
    st.session_state.pipeline_result = None
    st.session_state.itinerary_step_error = None

    audio_name = getattr(active_audio, "name", "recording.wav")
    audio_suffix = Path(audio_name).suffix.lower() or ".wav"
    tmp_audio_path = _save_uploaded_to_temp(active_audio, audio_suffix)

    try:
        transcript = transcribe_audio(
            tmp_audio_path,
            device=whisper_dev,
            model_name=DEFAULT_WHISPER_MODEL,
        )
        st.session_state.transcript = transcript
        _refresh_audio_direct_answer()
    finally:
        try:
            Path(tmp_audio_path).unlink(missing_ok=True)
        except OSError:
            pass
        clear_whisper_cache()
        gc.collect()


def _run_itinerary_step(
    check_time: datetime,
    db_path: str = DEFAULT_DB_PATH,
) -> None:
    """3. lépés: Kérdés megválaszolása + útiterv generálása 0.5B modellel, majd takarítás."""
    st.session_state.itinerary_step_error = None

    image_check: ImageCheckResult | None = st.session_state.image_check
    if image_check is None or st.session_state.coords is None:
        st.session_state.itinerary_step_error = (
            "Előbb futtasd le az 1. lépést (Kép ellenőrzése)!"
        )
        return

    matched_place = st.session_state.matched_place
    origin_coords = st.session_state.origin_coords or st.session_state.coords
    transcript = st.session_state.transcript
    time_label = check_time.strftime("%H:%M")

    category: str | None = matched_place["category"] if matched_place else None
    place_name = (
        matched_place["name"]
        if matched_place
        else (image_check.detected_name or "Ismeretlen látványosság")
    )

    same_category_places = _collect_db_candidates(
        category=category,
        origin_coords=origin_coords,
        exclude_id=matched_place["id"] if matched_place else None,
        exclude_name=place_name,
        db_path=db_path,
    )

    candidate_lines: list[str] = []
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
        built_str = format_built_year(matched_place.get("built_year"))
        context_parts.append(
            f"Épült: {built_str}, kategória: {matched_place['category']}, "
            f"nyitvatartás: {matched_place['opening_hours']}, jegyár: {matched_place['ticket_price_huf']} Ft"
        )
        if matched_place.get("description"):
            context_parts.append(f"Leírás: {matched_place['description']}")
    if transcript and transcript.get("text"):
        context_parts.append(f"Felhasználói kérdés a hangból: '{transcript['text']}'")
    if candidate_lines:
        context_parts.append("Elérhető adatbázis-megállók:\n" + "\n".join(candidate_lines))

    system_prompt = (
        "Magyar idegenvezető asszisztens vagy. Ne csak ismételd a hangot, hanem "
        "válaszolj közvetlenül a felhasználó kérdésére a helyszín adatai alapján (pl. évszám, ország, nyitvatartás, belépő)! "
        f"A 'felismert_hely' mezőbe MINDIG pontosan ezt írd: '{place_name}'. "
        "Az 'ajanlott_kovetkezo_megallo' mezőbe KIZÁRÓLAG az Elérhető adatbázis-megállók "
        "közül válassz egy pontos nevet, ne találj ki új helyet! "
        "Csak egy érvényes JSON objektumot adj vissza a következő kulcsokkal: "
        "felismert_hely, tortenelem_roviden, ajanlott_kovetkezo_megallo, "
        "becsult_ido_percben, megjegyzes."
    )

    closest_instruction = ""
    if same_category_places and "distance_km" in same_category_places[0]:
        closest_place = same_category_places[0]
        closest_instruction = (
            f"\nLegközelebbi adatbázis-megálló: '{closest_place['name']}', "
            f"becsult_ido_percben: {closest_place['estimated_walk_minutes']}."
        )

    user_prompt = (
        "\n".join(context_parts)
        + closest_instruction
        + f"\nFontos: felismert_hely = '{place_name}'. Válaszolj a kérdésre a 'megjegyzes' mezőben, "
        "és add meg a JSON mezőket magyarul."
    )

    itinerary_messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]

    try:
        try:
            tourist_response, is_open, status_msg, attempts = (
                generate_tourist_response_with_opening_check(
                    messages=itinerary_messages,
                    category=category,
                    current_place_name=place_name,
                    db_path=db_path,
                    current_time=check_time,
                    max_retries=1,
                    max_json_retries=1,
                    origin_coords=origin_coords,
                )
            )
        except StructuredOutputError:
            fallback_next = _pick_next_db_place(
                suggested_name=None,
                candidates=same_category_places,
                current_place_name=matched_place["name"] if matched_place else place_name,
                db_path=db_path,
            )
            tourist_response = _build_fallback_tourist_response(
                place_name=place_name,
                matched_place=matched_place,
                chosen_next=fallback_next,
                transcript=transcript,
            )
            if fallback_next and fallback_next.get("name"):
                is_open, status_msg = is_place_open(
                    fallback_next["name"],
                    current_time=check_time,
                    db_path=db_path,
                )
            else:
                is_open = False
                status_msg = "Nincs elérhető következő adatbázis-megálló."
            attempts = 2

        updates: dict[str, object] = {}
        if matched_place and matched_place.get("name"):
            updates["felismert_hely"] = matched_place["name"]

        chosen_next = _pick_next_db_place(
            suggested_name=tourist_response.ajanlott_kovetkezo_megallo,
            candidates=same_category_places,
            current_place_name=matched_place["name"] if matched_place else place_name,
            db_path=db_path,
        )
        if chosen_next is not None:
            updates["ajanlott_kovetkezo_megallo"] = chosen_next["name"]
            if chosen_next.get("estimated_walk_minutes"):
                updates["becsult_ido_percben"] = int(chosen_next["estimated_walk_minutes"])

        if updates:
            tourist_response = tourist_response.model_copy(update=updates)

        next_stop_distance_km: float | None = None
        next_place = find_place(
            tourist_response.ajanlott_kovetkezo_megallo,
            db_path=db_path,
        )
        if origin_coords is not None and next_place is not None:
            next_stop_distance_km = round(
                haversine_distance(
                    origin_coords[0],
                    origin_coords[1],
                    next_place["lat"],
                    next_place["lon"],
                ),
                3,
            )

        st.session_state.pipeline_result = PipelineResult(
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
    finally:
        clear_text_llm_cache()
        gc.collect()


def main() -> None:
    init_db_once(DEFAULT_DB_PATH)
    _init_session_state()

    st.title("📍 Multimodális Turisztikai Asszisztens (Lépésenkénti mód)")
    st.caption(
        "A memória kímélése érdekében a képellenőrzés, a hangátírás és az útiterv-készítés "
        "külön lépésekben fut le, és minden lépés után üríti a gyorsítótárakat."
    )

    with st.sidebar:
        st.header("Beállítások")
        use_custom_time = st.checkbox("Teszt időpont megadása", value=False)
        custom_time_str = st.text_input(
            "Időpont (HH:MM)",
            value=datetime.now().strftime("%H:%M"),
            disabled=not use_custom_time,
        )
        whisper_device_opt = st.selectbox(
            "Whisper eszköz",
            options=["auto", "cpu", "cuda"],
            index=0,
        )
        enable_tts = st.checkbox("Hangválasz generálása (TTS)", value=True)
        show_diagnostics = st.checkbox("Rendszerdiagnosztika mutatása", value=True)

        st.divider()
        if st.button("Állapot és gyorsítótárak törlése"):
            _reset_pipeline_state()
            st.rerun()

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("1. Kép (GPS nem kötelező)")
        uploaded_image = st.file_uploader(
            "Fotó a helyszínről (EXIF GPS nélkül is működik)",
            type=["jpg", "jpeg", "png", "webp"],
        )
        if uploaded_image is not None:
            st.image(uploaded_image, caption="Feltöltött fotó", width="stretch")

        if st.button("1. Kép ellenőrzése", type="primary", disabled=(uploaded_image is None)):
            with st.spinner("Kép ellenőrzése (CLIP)..."):
                try:
                    _run_image_step(uploaded_image, db_path=DEFAULT_DB_PATH)
                except Exception as exc:
                    st.session_state.image_step_error = f"Képellenőrzési hiba: {exc}"

        if st.session_state.image_step_error:
            st.error(st.session_state.image_step_error)
        elif st.session_state.image_check is not None:
            ic: ImageCheckResult = st.session_state.image_check
            place_label = (
                st.session_state.matched_place["name"]
                if st.session_state.matched_place
                else (ic.detected_name or "Ismeretlen helyszín")
            )
            st.success(
                f"Kép kész: **{place_label}** (megbízhatóság: {ic.confidence:.0%})"
            )
            if st.session_state.used_fallback_coords:
                st.info(
                    "Nem volt EXIF GPS a képen, ezért a helyszín/alapértelmezett koordinátát használtam."
                )

    with col2:
        st.subheader("2. Hangüzenet (kérdés és azonnali válasz)")
        uploaded_audio = st.file_uploader(
            "Hangüzenet (.wav, .mp3, .m4a)",
            type=["wav", "mp3", "m4a"],
        )
        recorded_audio = st.audio_input("Vagy rögzíts hangüzenetet mikrofonnal")
        active_audio = recorded_audio if recorded_audio is not None else uploaded_audio
        if active_audio is not None:
            st.audio(active_audio)

        whisper_dev = None if whisper_device_opt == "auto" else whisper_device_opt
        if st.button("2. Hang feldolgozása és válasz", disabled=(active_audio is None)):
            with st.spinner("Hang átírása (Whisper tiny) és válasz keresése..."):
                try:
                    _run_audio_step(active_audio, whisper_dev=whisper_dev)
                except AudioProcessingError as exc:
                    st.session_state.audio_step_error = f"Hangfeldolgozási hiba: {exc}"
                except Exception as exc:
                    st.session_state.audio_step_error = f"Váratlan hanghiba: {exc}"

        if st.session_state.audio_step_error:
            st.error(st.session_state.audio_step_error)
        elif st.session_state.transcript is not None:
            t_text = st.session_state.transcript.get("text") or "(üres leirat)"
            st.caption(f"Felismert kérdés: *„{t_text}”*")
            if st.session_state.audio_direct_answer:
                st.success(f"**Válasz:** {st.session_state.audio_direct_answer}")
                if enable_tts:
                    quick_spoken = st.session_state.audio_direct_answer.replace("**", "").replace("*", "")
                    quick_mp3 = synthesize_tts_mp3(quick_spoken, lang="hu")
                    if quick_mp3:
                        st.audio(quick_mp3, format="audio/mp3")

    st.divider()
    st.subheader("3. Útiterv és következő megálló")

    check_time: datetime = datetime.now()
    if use_custom_time:
        try:
            parsed = datetime.strptime(custom_time_str.strip(), "%H:%M")
            check_time = datetime.now().replace(
                hour=parsed.hour,
                minute=parsed.minute,
                second=0,
                microsecond=0,
            )
        except ValueError:
            st.warning("Érvénytelen HH:MM időformátum, az aktuális rendszeridőt használom.")

    can_run_step3 = (
        st.session_state.image_check is not None
        and st.session_state.coords is not None
    )
    if st.button("3. Útiterv generálása", type="primary", disabled=(not can_run_step3)):
        with st.spinner("Válasz és útiterv generálása (Qwen2.5-0.5B)..."):
            try:
                _run_itinerary_step(check_time=check_time, db_path=DEFAULT_DB_PATH)
            except (StructuredOutputError, ValueError) as exc:
                st.session_state.itinerary_step_error = f"Generálási hiba: {exc}"
            except Exception as exc:
                st.session_state.itinerary_step_error = f"Váratlan hiba: {exc}"

    if st.session_state.itinerary_step_error:
        st.error(st.session_state.itinerary_step_error)

    result: PipelineResult | None = st.session_state.pipeline_result
    if result is not None and result.success and result.tourist_response is not None:
        tr: TouristResponse = result.tourist_response
        unified_text = build_unified_response_text(result, db_path=DEFAULT_DB_PATH)

        st.subheader("🗺️ Összefoglaló és következő célpont")
        st.markdown(unified_text)

        if enable_tts:
            st.subheader("🔊 Hangválasz (TTS)")
            spoken_text = unified_text.replace("**", "").replace("*", "")
            mp3_bytes = synthesize_tts_mp3(spoken_text, lang="hu")
            if mp3_bytes:
                st.audio(mp3_bytes, format="audio/mp3")

        matched_place = result.matched_place or find_place(
            tr.felismert_hely,
            db_path=DEFAULT_DB_PATH,
        )
        next_place = find_place(
            tr.ajanlott_kovetkezo_megallo,
            db_path=DEFAULT_DB_PATH,
        )

        st.subheader("📌 Térkép (Folium)")
        folium_map = build_folium_map(
            origin_coords=st.session_state.origin_coords,
            matched_place=matched_place,
            next_place=next_place,
        )
        components.html(folium_map._repr_html_(), height=460, scrolling=False)

        if show_diagnostics:
            with st.expander("Rendszerdiagnosztika"):
                st.write(
                    {
                        "coords": (
                            {
                                "lat": st.session_state.coords[0],
                                "lon": st.session_state.coords[1],
                            }
                            if st.session_state.coords
                            else None
                        ),
                        "used_fallback_coords": st.session_state.used_fallback_coords,
                        "transcript": result.transcript,
                        "audio_direct_answer": st.session_state.audio_direct_answer,
                        "felismert_hely": tr.felismert_hely,
                        "felismert_evszam": (
                            matched_place.get("built_year") if matched_place else None
                        ),
                        "ajanlott_kovetkezo_megallo": tr.ajanlott_kovetkezo_megallo,
                        "ajanlott_evszam": (
                            next_place.get("built_year") if next_place else None
                        ),
                        "is_next_stop_open": result.is_next_stop_open,
                        "next_stop_distance_km": result.next_stop_distance_km,
                        "attempts": result.attempts,
                    }
                )


if __name__ == "__main__":
    main()