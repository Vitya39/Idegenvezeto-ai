from PIL import ExifTags, Image


def _to_float(val) -> float:
    """Safely convert an EXIF IFDRational or (numerator, denominator) tuple to float."""
    if isinstance(val, tuple) and len(val) == 2:
        return float(val[0]) / float(val[1]) if val[1] != 0 else 0.0
    return float(val)


def _dms_to_decimal(dms, ref: str) -> float:
    """Konvertálja az EXIF-ből kiolvasott szög-szögperc-szögmásodperc (DMS) koordinátákat decimálissá."""
    szog = _to_float(dms[0])
    szogperc = _to_float(dms[1])
    szogmasodperc = _to_float(dms[2])
    decimal = szog + (szogperc / 60.0) + (szogmasodperc / 3600.0)
    if ref in ("S", "W", b"S", b"W"):
        decimal = -decimal
    return decimal


def get_image_coordinates(
    image_source: str | Image.Image,
) -> tuple[float, float] | None:
    """Kivonja egy kép EXIF GPS koordinátáit (szélesség, hosszúság) tizedes formátumban."""
    if image_source is None:
        return None

    image = Image.open(image_source) if isinstance(image_source, str) else image_source
    exif = image.getexif()
    if not exif:
        return None

    gps_ifd = exif.get_ifd(ExifTags.IFD.GPSInfo)
    if not gps_ifd:
        raw_exif = getattr(image, "_getexif", lambda: None)()
        if raw_exif:
            for tag_id, value in raw_exif.items():
                if ExifTags.TAGS.get(tag_id) == "GPSInfo":
                    gps_ifd = value
                    break

    if not gps_ifd:
        return None

    gps_data = {ExifTags.GPSTAGS.get(key, key): val for key, val in gps_ifd.items()}

    lat_dms = gps_data.get("GPSLatitude")
    lat_ref = gps_data.get("GPSLatitudeRef", "N")
    lon_dms = gps_data.get("GPSLongitude")
    lon_ref = gps_data.get("GPSLongitudeRef", "E")

    if lat_dms and lon_dms:
        lat = _dms_to_decimal(lat_dms, lat_ref)
        lon = _dms_to_decimal(lon_dms, lon_ref)
        return lat, lon

    return None
