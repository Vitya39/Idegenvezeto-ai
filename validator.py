"""SQLite adatbázis-lekérdezések és nyitvatartás-ellenőrzés a tourism.db alapján."""

from __future__ import annotations

from collections.abc import Iterable
from datetime import datetime
import math
from pathlib import Path
import re
import sqlite3
from typing import Any
import unicodedata

from database import create_and_populate_db

DEFAULT_DB_PATH: str = "tourism.db"
_TIME_PATTERN = re.compile(r"^(\d{1,2}):(\d{2})$")


def _normalize_text(text: str) -> str:
    """Kisbetűs, ékezetmentesített és egységesített szöveg összehasonlításhoz."""
    decomposed = unicodedata.normalize("NFKD", text.strip().casefold())
    without_accents = "".join(
        ch for ch in decomposed if unicodedata.category(ch) != "Mn"
    )
    unified_separators = without_accents.replace("_", " ").replace("-", " ")
    return " ".join(unified_separators.split())


def _connect_db(db_path: str = DEFAULT_DB_PATH) -> sqlite3.Connection:
    """Megnyitja az SQLite kapcsolatot, alapértelmezett DB hiánya esetén létrehozza azt."""
    path = Path(db_path)
    if not path.exists():
        if db_path == DEFAULT_DB_PATH:
            create_and_populate_db(db_path)
        else:
            raise FileNotFoundError(f"Az adatbázis fájl nem található: {db_path}")

    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    return conn


def get_all_places(db_path: str = DEFAULT_DB_PATH) -> list[dict[str, Any]]:
    """Visszaadja az összes helyszínt a places táblából."""
    conn = _connect_db(db_path)
    try:
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM places")
        return [dict(row) for row in cursor.fetchall()]
    finally:
        conn.close()


def find_place(query: str, db_path: str = DEFAULT_DB_PATH) -> dict[str, Any] | None:
    """Helyszín keresése név vagy id alapján (kis-/nagybetű-, ékezet- és részleges egyezéssel)."""
    if not query or not query.strip():
        return None

    places = get_all_places(db_path)
    query_cf = query.strip().casefold()
    query_norm = _normalize_text(query)

    # 1. Pontos egyezés (kis-/nagybetű függetlenül) id vagy név alapján
    for place in places:
        if place["id"].casefold() == query_cf or place["name"].casefold() == query_cf:
            return place

    # 2. Pontos egyezés ékezet- és elválasztó-normalizálás után
    for place in places:
        if (
            _normalize_text(place["id"]) == query_norm
            or _normalize_text(place["name"]) == query_norm
        ):
            return place

    # 3. Részleges (tartalmazásos) egyezés, a legközelebbi hosszúságú találat előnyben
    partial_matches: list[tuple[int, int, dict[str, Any]]] = []
    for place in places:
        id_norm = _normalize_text(place["id"])
        name_norm = _normalize_text(place["name"])
        if (
            query_norm in name_norm
            or query_norm in id_norm
            or (len(name_norm) >= 3 and name_norm in query_norm)
            or (len(id_norm) >= 3 and id_norm in query_norm)
        ):
            length_diff = min(
                abs(len(name_norm) - len(query_norm)),
                abs(len(id_norm) - len(query_norm)),
            )
            partial_matches.append((length_diff, len(name_norm), place))

    if partial_matches:
        partial_matches.sort(key=lambda item: (item[0], item[1]))
        return partial_matches[0][2]

    return None


def get_places_by_category(
    category: str,
    db_path: str = DEFAULT_DB_PATH,
    exclude_name: str | None = None,
    exclude_id: str | None = None,
) -> list[dict[str, Any]]:
    """Azonos kategóriába tartozó helyszínek listázása, opcionális kizárással."""
    if not category or not category.strip():
        return []

    target_cat = _normalize_text(category)
    excl_name_norm = _normalize_text(exclude_name) if exclude_name else None
    excl_id_norm = _normalize_text(exclude_id) if exclude_id else None

    results: list[dict[str, Any]] = []
    for place in get_all_places(db_path):
        if _normalize_text(place["category"]) != target_cat:
            continue
        if excl_id_norm and _normalize_text(place["id"]) == excl_id_norm:
            continue
        if excl_name_norm and _normalize_text(place["name"]) == excl_name_norm:
            continue
        results.append(place)

    return results


def haversine_distance(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Két földrajzi koordináta légvonalbeli távolsága kilométerben (Haversine-képlet)."""
    earth_radius_km = 6371.0
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lon2 - lon1)

    a = (
        math.sin(dphi / 2.0) ** 2
        + math.cos(phi1) * math.cos(phi2) * math.sin(dlambda / 2.0) ** 2
    )
    return 2.0 * earth_radius_km * math.atan2(math.sqrt(a), math.sqrt(1.0 - a))


def estimate_walk_minutes(distance_km: float, speed_kmh: float = 5.0) -> int:
    """Megbecsüli a gyalogos eljutási időt percben a távolság alapján."""
    return max(1, round((distance_km / speed_kmh) * 60.0))


def get_places_by_category_sorted_by_distance(
    category: str,
    origin_lat: float,
    origin_lon: float,
    db_path: str = DEFAULT_DB_PATH,
    exclude_name: str | None = None,
    exclude_id: str | None = None,
) -> list[dict[str, Any]]:
    """Azonos kategóriába tartozó helyszínek listázása szigorúan távolság szerint rendezve."""
    places = get_places_by_category(
        category,
        db_path=db_path,
        exclude_name=exclude_name,
        exclude_id=exclude_id,
    )
    enriched_places: list[dict[str, Any]] = []
    for place in places:
        dist_km = haversine_distance(origin_lat, origin_lon, place["lat"], place["lon"])
        enriched = dict(place)
        enriched["distance_km"] = round(dist_km, 3)
        enriched["estimated_walk_minutes"] = estimate_walk_minutes(dist_km)
        enriched_places.append(enriched)

    enriched_places.sort(key=lambda item: item["distance_km"])
    return enriched_places


def _parse_hhmm_to_minutes(time_str: str, is_end: bool = False) -> int:
    """HH:MM időbélyeg percben történő értelmezése a 24:00 és záró 00:00 kezelésével."""
    match = _TIME_PATTERN.match(time_str.strip())
    if not match:
        raise ValueError(f"Érvénytelen időformátum (várt HH:MM): {time_str!r}")

    hour = int(match.group(1))
    minute = int(match.group(2))

    if hour == 24 and minute == 0:
        return 24 * 60

    if not (0 <= hour <= 23 and 0 <= minute <= 59):
        raise ValueError(f"Érvénytelen óra/perc érték: {time_str!r}")

    if is_end and hour == 0 and minute == 0:
        return 24 * 60

    return hour * 60 + minute


def is_opening_hours_open(
    opening_hours: str,
    current_time: datetime | None = None,
) -> bool:
    """Kiértékeli a 'HH:MM-HH:MM' nyitvatartási intervallumot a megadott időpontra."""
    if not opening_hours or "-" not in opening_hours:
        raise ValueError(f"Érvénytelen nyitvatartási formátum: {opening_hours!r}")

    parts = [part.strip() for part in opening_hours.strip().split("-")]
    if len(parts) != 2 or not parts[0] or not parts[1]:
        raise ValueError(f"Érvénytelen nyitvatartási formátum: {opening_hours!r}")

    open_min = _parse_hhmm_to_minutes(parts[0], is_end=False)
    close_min = _parse_hhmm_to_minutes(parts[1], is_end=True)

    check_time = current_time or datetime.now()
    now_min = check_time.hour * 60 + check_time.minute

    if open_min == 0 and close_min == 24 * 60:
        return True
    if open_min < close_min:
        return open_min <= now_min < close_min
    if open_min > close_min:
        return now_min >= open_min or now_min < close_min
    return False


def is_place_open(
    place_name: str,
    db_path: str = DEFAULT_DB_PATH,
    current_time: datetime | None = None,
) -> tuple[bool, str]:
    """Ellenőrzi, hogy a megadott helyszín nyitva tart-e az aktuális vagy teszt időpontban."""
    check_time = current_time or datetime.now()
    time_label = check_time.strftime("%H:%M")

    place = find_place(place_name, db_path=db_path)
    if place is None:
        return (
            False,
            f"A(z) '{place_name}' helyszín nem található az adatbázisban.",
        )

    opening_hours = place["opening_hours"]
    try:
        is_open = is_opening_hours_open(opening_hours, current_time=check_time)
    except ValueError as exc:
        return (
            False,
            f"A(z) {place['name']} nyitvatartási adata hibás ({opening_hours}): {exc}",
        )

    if is_open:
        return (
            True,
            f"A(z) {place['name']} jelenleg nyitva tart ({time_label}; nyitvatartás: {opening_hours}).",
        )
    return (
        False,
        f"A(z) {place['name']} jelenleg zárva van ({time_label}; nyitvatartás: {opening_hours}).",
    )


def get_open_places_in_category(
    category: str,
    db_path: str = DEFAULT_DB_PATH,
    current_time: datetime | None = None,
    exclude_names: Iterable[str] | None = None,
    origin_coords: tuple[float, float] | None = None,
) -> list[dict[str, Any]]:
    """Visszaadja az adott kategórián belül jelenleg nyitva tartó helyszíneket (távolság szerint rendezve)."""
    check_time = current_time or datetime.now()
    excluded = {
        _normalize_text(name)
        for name in (exclude_names or ())
        if name and name.strip()
    }

    open_places: list[dict[str, Any]] = []
    for place in get_places_by_category(category, db_path=db_path):
        if (
            _normalize_text(place["name"]) in excluded
            or _normalize_text(place["id"]) in excluded
        ):
            continue
        try:
            if is_opening_hours_open(place["opening_hours"], current_time=check_time):
                item = dict(place)
                if origin_coords is not None:
                    dist_km = haversine_distance(
                        origin_coords[0],
                        origin_coords[1],
                        place["lat"],
                        place["lon"],
                    )
                    item["distance_km"] = round(dist_km, 3)
                    item["estimated_walk_minutes"] = estimate_walk_minutes(dist_km)
                open_places.append(item)
        except ValueError:
            continue

    if origin_coords is not None:
        open_places.sort(key=lambda item: item["distance_km"])

    return open_places


def build_closed_place_feedback(
    place_name: str,
    open_places_in_category: list[str],
) -> str:
    """Összeállítja az LLM visszacsatolási hurok rendszerüzenetét zárt helyszín esetén."""
    places_str = ", ".join(open_places_in_category)
    return (
        f"A javasolt hely ({place_name}) jelenleg zárva van. "
        f"Válassz egy másik, jelenleg nyitva tartó látványosságot ebből a listából: "
        f"{places_str}"
    )
