from datetime import datetime
import math
import sqlite3
import time
import warnings
import gradio as gr
import librosa
from PIL import ExifTags, Image
from qwen_vl_utils import process_vision_info
import torch
from transformers import (
    AutoModelForCausalLM,
    AutoProcessor,
    AutoTokenizer,
    BitsAndBytesConfig,
    Qwen2_5_VLForConditionalGeneration,
    logging as hf_logging,
    pipeline,
)
from database import create_and_populate_db

# Felesleges transformers belső figyelmeztetések elnémítása
warnings.filterwarnings("ignore", category=UserWarning)
hf_logging.set_verbosity_error()

# Adatbázis biztosítása induláskor
DB_PATH = "tourism.db"
create_and_populate_db()

# Eszköz és 4-bites kvantálás beállítása (hogy mind a 3 modell kényelmesen, <4 GB VRAM-ban elférjen)
has_cuda = torch.cuda.is_available()
device_str = "cuda" if has_cuda else "cpu"
model_dtype = torch.float16 if has_cuda else torch.float32
print(f"Futtatási környezet: {device_str.upper()} (dtype: {model_dtype})")

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

# Képfelbontás korlátozása a gyors és memóriatakarékos képfelismeréshez
VISION_MODEL_ID = "Qwen/Qwen2.5-VL-3B-Instruct"
MIN_PIXELS = 256 * 28 * 28
MAX_PIXELS = 512 * 28 * 28

# 1. Modell: Képfelismerő modell (<= 3B paraméter: Qwen2.5-VL-3B-Instruct, 4-biten ~2.3 GB VRAM)
print("1/3 Képmodell (Qwen2.5-VL-3B-Instruct) betöltése...")
vision_model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    VISION_MODEL_ID,
    quantization_config=bnb_config,
    dtype=model_dtype,
    device_map="auto" if has_cuda else None,
    low_cpu_mem_usage=True,
)
vision_processor = AutoProcessor.from_pretrained(
    VISION_MODEL_ID,
    min_pixels=MIN_PIXELS,
    max_pixels=MAX_PIXELS,
)

# 2. Modell: Hangfelismerő modell (Whisper Small, fp16-ban ~0.5 GB VRAM)
print("2/3 Hangmodell (openai/whisper-small) betöltése...")
asr_pipe = pipeline(
    "automatic-speech-recognition",
    model="openai/whisper-small",
    dtype=model_dtype,
    device=0 if has_cuda else -1,
)

# 3. Modell: Nyelvi modell (<= 3B paraméter: Qwen2.5-1.5B-Instruct, 4-biten ~1.1 GB VRAM)
print("3/3 Nyelvi modell (Qwen2.5-1.5B-Instruct) betöltése...")
LLM_MODEL_ID = "Qwen/Qwen2.5-1.5B-Instruct"
llm_tokenizer = AutoTokenizer.from_pretrained(LLM_MODEL_ID)
llm_model = AutoModelForCausalLM.from_pretrained(
    LLM_MODEL_ID,
    quantization_config=bnb_config,
    dtype=model_dtype,
    device_map="auto" if has_cuda else None,
    low_cpu_mem_usage=True,
)
llm_pipe = pipeline(
    "text-generation",
    model=llm_model,
    tokenizer=llm_tokenizer,
)

if has_cuda:
    allocated_gb = torch.cuda.memory_allocated(0) / 1e9
    print(f"Minden modell készen áll! Foglalt VRAM: {allocated_gb:.2f} GB")
else:
    print("Minden modell készen áll!")


def lekerdez_helyszinek(db_path: str = DB_PATH) -> list[dict]:
    """Beolvassa az összes helyszínt a tourism.db adatbázisból."""
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM places")
    rows = [dict(row) for row in cursor.fetchall()]
    conn.close()
    return rows


def _to_float(val) -> float:
    """EXIF IFDRational vagy tuple érték biztonságos float-tá alakítása."""
    if isinstance(val, tuple) and len(val) == 2:
        return float(val[0]) / float(val[1]) if val[1] != 0 else 0.0
    return float(val)


def _dms_to_decimal(dms, ref: str) -> float:
    """EXIF fok-perc-másodperc (DMS) konvertálása decimális koordinátává."""
    degrees = _to_float(dms[0])
    minutes = _to_float(dms[1])
    seconds = _to_float(dms[2])
    decimal = degrees + (minutes / 60.0) + (seconds / 3600.0)
    if ref in ("S", "W", b"S", b"W"):
        decimal = -decimal
    return decimal


def kinyer_kep_koordinatak(
    kep_forras: str | Image.Image,
) -> tuple[float, float] | None:
    """Külön függvény a feltöltött kép EXIF GPS koordinátáinak kinyerésére."""
    if kep_forras is None:
        return None
    try:
        kep = (
            Image.open(kep_forras)
            if isinstance(kep_forras, str)
            else kep_forras
        )
        exif = kep.getexif()
        if not exif:
            return None

        gps_ifd = exif.get_ifd(ExifTags.IFD.GPSInfo)
        if not gps_ifd:
            raw_exif = getattr(kep, "_getexif", lambda: None)()
            if raw_exif:
                for tag_id, value in raw_exif.items():
                    if ExifTags.TAGS.get(tag_id) == "GPSInfo":
                        gps_ifd = value
                        break

        if not gps_ifd:
            return None

        gps_data = {}
        for key, val in gps_ifd.items():
            decoded = ExifTags.GPSTAGS.get(key, key)
            gps_data[decoded] = val

        lat_dms = gps_data.get("GPSLatitude")
        lat_ref = gps_data.get("GPSLatitudeRef", "N")
        lon_dms = gps_data.get("GPSLongitude")
        lon_ref = gps_data.get("GPSLongitudeRef", "E")

        if lat_dms and lon_dms:
            lat = _dms_to_decimal(lat_dms, lat_ref)
            lon = _dms_to_decimal(lon_dms, lon_ref)
            return (lat, lon)
    except Exception:
        return None

    return None


def haversine_tavolsag(
    lat1: float, lon1: float, lat2: float, lon2: float
) -> float:
    """Két földrajzi koordináta távolsága kilométerben (Haversine-képlet)."""
    r = 6371.0
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lon2 - lon1)

    a = (
        math.sin(dphi / 2.0) ** 2
        + math.cos(phi1) * math.cos(phi2) * math.sin(dlambda / 2.0) ** 2
    )
    return 2.0 * r * math.atan2(math.sqrt(a), math.sqrt(1.0 - a))


def nyitva_van_e(opening_hours: str, aktualis_ido: datetime) -> bool:
    """Ellenőrzi, hogy az adott helyszín nyitva van-e a rendszeridő alapján."""
    try:
        open_str, close_str = opening_hours.strip().split("-")
        open_h, open_m = map(int, open_str.split(":"))
        close_h, close_m = map(int, close_str.split(":"))

        open_minutes = open_h * 60 + open_m
        close_minutes = close_h * 60 + close_m
        if close_minutes == 0 or close_h == 24:
            close_minutes = 24 * 60

        now_minutes = aktualis_ido.hour * 60 + aktualis_ido.minute

        if open_minutes <= close_minutes:
            return open_minutes <= now_minutes <= close_minutes
        else:
            return now_minutes >= open_minutes or now_minutes <= close_minutes
    except Exception:
        return False


def felismer_helyszin_keprol(
    kep: Image.Image, helyszinek: list[dict]
) -> tuple[dict | None, str]:
    """1. Modell (Qwen2.5-VL-3B): Felismeri a fotón látható objektumot és kiválasztja a tourism.db-ből."""
    helyszin_lista_szoveg = "\n".join(
        [f"- {h['id']}: {h['name']} ({h['category']})" for h in helyszinek]
    )

    prompt = (
        "Nézd meg a képet, és döntsd el, hogy az alábbi adatbázis-helyszínek közül "
        "melyik látható rajta!\n"
        "Választható helyszínek:\n"
        f"{helyszin_lista_szoveg}\n\n"
        "Válaszként CSAK a felismert helyszín pontos 'id' azonosítóját és nevét írd le "
        "ebben a formátumban: ID: <id> | NÉV: <name>"
    )

    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "image",
                    "image": kep,
                    "min_pixels": MIN_PIXELS,
                    "max_pixels": MAX_PIXELS,
                },
                {"type": "text", "text": prompt},
            ],
        }
    ]

    text = vision_processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    image_inputs, video_inputs = process_vision_info(messages)

    target_device = "cuda" if has_cuda else "cpu"
    inputs = vision_processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    ).to(target_device)

    with torch.inference_mode():
        generated_ids = vision_model.generate(
            **inputs, max_new_tokens=40, do_sample=False
        )

    generated_ids_trimmed = [
        out_ids[len(in_ids) :]
        for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]
    valasz_szoveg = vision_processor.batch_decode(
        generated_ids_trimmed,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )[0].strip()

    valasz_lower = valasz_szoveg.lower()
    for h in helyszinek:
        if h["id"].lower() in valasz_lower or h["name"].lower() in valasz_lower:
            return h, valasz_szoveg

    return None, valasz_szoveg


def helyszin_es_ajanlo(kep_utvonal, hang_fajl):
    try:
        start_total = time.time()
        if kep_utvonal is None and hang_fajl is None:
            return (
                "Kérlek tölts fel egy fotót (GPS metaadattal) és adj meg egy hangüzenetet!",
                "Nincs bemenet.",
            )
        if kep_utvonal is None:
            return (
                "Kérlek tölts fel egy fotót a helyszín felismeréséhez!",
                "Hiányzó fotó.",
            )
        if hang_fajl is None:
            return (
                "Kérlek tölts fel vagy vegyél fel egy hangüzenetet!",
                "Hiányzó hangüzenet.",
            )

        # Kötelező EXIF GPS koordináta ellenőrzés külön függvénnyel
        exif_koordinatak = kinyer_kep_koordinatak(kep_utvonal)
        if exif_koordinatak is None:
            return (
                "Hiba: A feltöltött kép nem tartalmaz GPS (EXIF) helyadatokat! "
                "A távolságszámításhoz kötelező olyan fotót feltölteni, amely tartalmaz helymeghatározási metaadatot.",
                "❌ Hiányzó EXIF GPS metaadat a feltöltött képen.",
            )

        user_lat, user_lon = exif_koordinatak
        koordinata_forras = (
            f"Kép EXIF GPS metaadata ({user_lat:.5f}, {user_lon:.5f})"
        )

        kep_pil = (
            Image.open(kep_utvonal).convert("RGB")
            if isinstance(kep_utvonal, str)
            else kep_utvonal.convert("RGB")
        )
        helyszinek = lekerdez_helyszinek()
        aktualis_ido = datetime.now()
        aktualis_ido_str = aktualis_ido.strftime("%H:%M")

        # 1. LÉPÉS: Kép felismerése az 1. modellel (Qwen2.5-VL-3B-Instruct)
        t0 = time.time()
        print("[1/3] Kép elemzése (Qwen2.5-VL-3B)...")
        felismert_helyszin, vision_nyers_valasz = felismer_helyszin_keprol(
            kep_pil, helyszinek
        )
        t_vision = time.time() - t0
        print(f"      Kész ({t_vision:.1f}s): {vision_nyers_valasz}")

        # 2. LÉPÉS: Hangüzenet leiratozása a 2. modellel (Whisper Small)
        t0 = time.time()
        print("[2/3] Hangüzenet leiratozása (Whisper)...")
        speech_array, sampling_rate = librosa.load(hang_fajl, sr=16000)
        hang_eredmeny = asr_pipe(
            {"array": speech_array, "sampling_rate": sampling_rate},
            generate_kwargs={"language": "hungarian", "task": "transcribe"},
        )
        hang_szoveg = hang_eredmeny["text"].strip()
        t_asr = time.time() - t0
        print(f"      Kész ({t_asr:.1f}s): '{hang_szoveg}'")

        # 3. LÉPÉS: Távolságok és rendszeridő alapú nyitvatartás számítása a kép GPS koordinátái alapján
        ertekelt_helyszinek = []
        for h in helyszinek:
            tavolsag_km = haversine_tavolsag(
                user_lat, user_lon, h["lat"], h["lon"]
            )
            nyitva = nyitva_van_e(h["opening_hours"], aktualis_ido)
            ertekelt_helyszinek.append(
                {
                    **h,
                    "distance_km": tavolsag_km,
                    "is_open": nyitva,
                }
            )

        nyitva_tarto_helyek = sorted(
            [h for h in ertekelt_helyszinek if h["is_open"]],
            key=lambda x: x["distance_km"],
        )

        mas_nyitva_tarto_helyek = [
            h
            for h in nyitva_tarto_helyek
            if not felismert_helyszin or h["id"] != felismert_helyszin["id"]
        ]

        legkozelebbi_nyitva = (
            nyitva_tarto_helyek[0] if nyitva_tarto_helyek else None
        )
        legkozelebbi_masik_nyitva = (
            mas_nyitva_tarto_helyek[0] if mas_nyitva_tarto_helyek else None
        )

        rendezett_osszes = sorted(
            ertekelt_helyszinek, key=lambda x: x["distance_km"]
        )
        helyszin_osszefoglalo_sorok = []
        for h in rendezett_osszes[:6]:
            statusz = "NYITVA" if h["is_open"] else "ZÁRVA"
            helyszin_osszefoglalo_sorok.append(
                f"- {h['name']} ({h['category']}): "
                f"{h['distance_km']:.2f} km | "
                f"Nyitvatartás: {h['opening_hours']} ({statusz}) | "
                f"Jegyár: {h['ticket_price_huf']} Ft | {h['description']}"
            )
        helyszin_osszefoglalo = "\n".join(helyszin_osszefoglalo_sorok)

        felismert_leiras = (
            f"{felismert_helyszin['name']} (ID: {felismert_helyszin['id']}, "
            f"nyitvatartás: {felismert_helyszin['opening_hours']}, "
            f"leírás: {felismert_helyszin['description']})"
            if felismert_helyszin
            else f"Nem sikerült egyértelműen azonosítani ({vision_nyers_valasz})"
        )

        legkozelebbi_info = (
            f"A rendszeridő ({aktualis_ido_str}) alapján a legközelebbi NYITVA tartó helyszín: "
            f"{legkozelebbi_nyitva['name']} ({legkozelebbi_nyitva['distance_km']:.2f} km, "
            f"nyitvatartás: {legkozelebbi_nyitva['opening_hours']})."
            if legkozelebbi_nyitva
            else f"A rendszeridő ({aktualis_ido_str}) alapján jelenleg egyetlen helyszín sincs nyitva."
        )
        if (
            legkozelebbi_masik_nyitva
            and legkozelebbi_nyitva
            and legkozelebbi_masik_nyitva["id"] != legkozelebbi_nyitva["id"]
        ):
            legkozelebbi_info += (
                f"\nA fotózott helyszínen kívüli legközelebbi NYITVA tartó helyszín: "
                f"{legkozelebbi_masik_nyitva['name']} ({legkozelebbi_masik_nyitva['distance_km']:.2f} km, "
                f"nyitvatartás: {legkozelebbi_masik_nyitva['opening_hours']}, "
                f"jegyár: {legkozelebbi_masik_nyitva['ticket_price_huf']} Ft)."
            )

        # 4. LÉPÉS: 3. Modell (LLM) elemzi a fotózott objektumot és a hangfelvételt, majd válaszol
        t0 = time.time()
        print("[3/3] Végső válasz generálása (Qwen2.5-1.5B)...")
        uzenetek = [
            {
                "role": "system",
                "content": (
                    "Te egy profi, barátságos magyar idegenvezető asszisztens vagy. "
                    "A feladatod, hogy az 1. modell által felismert fotózott objektum, "
                    "a 2. modell által leiratozott hangüzenet, valamint a kép GPS koordinátáiból "
                    "számított távolság és nyitvatartási adatok alapján pontosan válaszolj a turistának."
                ),
            },
            {
                "role": "user",
                "content": (
                    f"Aktuális rendszeridő: {aktualis_ido_str}\n"
                    f"1. A fotón felismert objektum (tourism.db-ből): {felismert_leiras}\n"
                    f"2. A turista hangüzenete: '{hang_szoveg}'\n"
                    f"3. Kép GPS koordinátái: {user_lat:.5f}, {user_lon:.5f}\n\n"
                    f"Kiszámított nyitvatartási és távolsági eredmény:\n{legkozelebbi_info}\n\n"
                    f"Legközelebbi helyszínek listája az adatbázisból:\n{helyszin_osszefoglalo}\n\n"
                    "Kérlek, válaszolj magyarul a turistának az alábbiak szerint:\n"
                    "1. Mondd el, melyik helyszínt ismertük fel a feltöltött fotón!\n"
                    "2. Reagálj a hangüzenetben elhangzott kérésre vagy kérdésre!\n"
                    "3. Pontosan nevezd meg, melyik helyszín van a legközelebb a places_data (tourism.db) "
                    "adatbázisból, amely az aktuális rendszeridő alapján NYITVA van (add meg a távolságát, "
                    "nyitvatartását és jegyárát is)!"
                ),
            },
        ]

        prompt_text = llm_pipe.tokenizer.apply_chat_template(
            uzenetek, tokenize=False, add_generation_prompt=True
        )
        with torch.inference_mode():
            llm_kimenet = llm_pipe(
                prompt_text,
                max_new_tokens=280,
                do_sample=True,
                temperature=0.3,
                return_full_text=False,
                pad_token_id=llm_pipe.tokenizer.eos_token_id,
            )
        vegso_valasz = llm_kimenet[0]["generated_text"].strip()
        t_llm = time.time() - t0
        total_sec = time.time() - start_total
        print(f"      Kész ({t_llm:.1f}s). Összesen: {total_sec:.1f}s")

        diagnosztika = (
            f"⏱️ Feldolgozási idő: {total_sec:.1f} mp (Kép: {t_vision:.1f}s, Hang: {t_asr:.1f}s, LLM: {t_llm:.1f}s)\n"
            f"🕒 Rendszeridő: {aktualis_ido_str}\n"
            f"📷 1. Modell (Qwen2.5-VL-3B) felismert helyszín: "
            f"{felismert_helyszin['name'] if felismert_helyszin else 'Nem azonosított'} "
            f"(Nyers: {vision_nyers_valasz})\n"
            f"🎤 2. Modell (Whisper) hangüzenet átirata: '{hang_szoveg}'\n"
            f"📍 Helymeghatározás forrása: {koordinata_forras}\n"
            f"✅ Legközelebbi nyitva tartó helyszín: {legkozelebbi_info}"
        )

        return vegso_valasz, diagnosztika

    except Exception as e:
        return f"Hiba történt: {str(e)}", f"Részletek: {str(e)}"


# Gradio felület (csak fotó és hangüzenet bemenettel)
demo = gr.Interface(
    fn=helyszin_es_ajanlo,
    inputs=[
        gr.Image(
            type="filepath",
            label="1. Fotó a helyszínről (Kötelező EXIF GPS helyadattal!)",
        ),
        gr.Audio(
            sources=["microphone", "upload"],
            type="filepath",
            label="2. Hangüzenet (Feltöltés vagy mikrofon felvétel)",
        ),
    ],
    outputs=[
        gr.Textbox(label="🌍 Idegenvezető Válasza (3. Modell)", lines=12),
        gr.Textbox(label="🔗 Rendszer Diagnosztika", lines=6),
    ],
    title="📍 Multimodális Turisztikai Asszisztens (3 Modelles Pipeline + SQLite)",
    description=(
        "Tölts fel egy GPS (EXIF) helyadatot tartalmazó fotót és vegyél fel / tölts fel egy hangüzenetet! "
        "Az 1. modell azonosítja a fotón lévő helyszínt a tourism.db-ből, "
        "a 2. modell leiratozza a hangüzeneted, a 3. modell pedig a kép GPS koordinátái "
        "és az aktuális rendszeridő alapján megmondja, melyik a legközelebbi nyitva tartó helyszín."
    ),
)

if __name__ == "__main__":
    demo.launch(inbrowser=True)
