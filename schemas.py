"""Pydantic v2 adatsémák a strukturált LLM válaszokhoz és a pipeline eredményéhez."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class ImageCheckResult(BaseModel):
    """A feltöltött kép látványosság-ellenőrzési (guardrail) eredménye."""

    model_config = ConfigDict(str_strip_whitespace=True)

    is_landmark: bool = Field(
        ...,
        description="Igaz, ha a képen turisztikai látványosság vagy nevezetesség látható.",
    )
    confidence: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="A felismerés megbízhatósága 0.0 és 1.0 között.",
    )
    detected_name: str | None = Field(
        default=None,
        description="A felismert látványosság neve, ha azonosítható.",
    )
    reason: str = Field(
        ...,
        description="Rövid magyar nyelvű indoklás a döntéshez.",
    )


class TouristResponse(BaseModel):
    """A turista számára generált strukturált idegenvezetői válasz sémája."""

    model_config = ConfigDict(str_strip_whitespace=True)

    felismert_hely: str = Field(
        ...,
        min_length=1,
        description="A képen felismert látványosság neve.",
    )
    tortenelem_roviden: str = Field(
        ...,
        min_length=1,
        description="Rövid történelmi és turisztikai összefoglaló.",
    )
    ajanlott_kovetkezo_megallo: str = Field(
        ...,
        min_length=1,
        description="A javasolt következő, nyitva tartó látványosság neve.",
    )
    becsult_ido_percben: int = Field(
        ...,
        ge=0,
        description="A következő megállóhoz becsült látogatási vagy eljutási idő percben.",
    )
    megjegyzes: str | None = Field(
        default=None,
        description="Opcionális kiegészítő megjegyzés vagy praktikus tipp.",
    )


class PipelineResult(BaseModel):
    """A teljes turisztikai asszisztens pipeline végrehajtási eredménye."""

    success: bool = Field(..., description="Igaz, ha a folyamat sikeresen lefutott.")
    image_check: ImageCheckResult = Field(
        ...,
        description="A bemeneti kép guardrail ellenőrzési eredménye.",
    )
    tourist_response: TouristResponse | None = Field(
        default=None,
        description="A validált turisztikai válasz, ha a guardrail átengedte.",
    )
    transcript: dict[str, str] | None = Field(
        default=None,
        description="A hangüzenet átirata és detektált nyelve, ha hangfájl is érkezett.",
    )
    matched_place: dict[str, Any] | None = Field(
        default=None,
        description="A tourism.db-ből beazonosított helyszín adatai.",
    )
    is_next_stop_open: bool | None = Field(
        default=None,
        description="Igaz, ha az ajánlott következő megálló nyitva tart.",
    )
    next_stop_distance_km: float | None = Field(
        default=None,
        description="A javasolt következő megálló távolsága kilométerben a kiindulási ponttól.",
    )
    opening_status_message: str | None = Field(
        default=None,
        description="Szöveges visszajelzés a nyitvatartás-ellenőrzés eredményéről.",
    )
    error_message: str | None = Field(
        default=None,
        description="Felhasználóbarát hibaüzenet elutasítás vagy hiba esetén.",
    )
    attempts: int = Field(
        default=0,
        ge=0,
        description="A generálási kísérletek száma a nyitvatartási hurokban.",
    )
