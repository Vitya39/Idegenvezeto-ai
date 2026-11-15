import gradio as gr
import librosa
import torch
from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info

# Eszköz beállítása (ha van Nvidia GPU-d, azt használja, különben CPU)
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Futtatási környezet: {device.upper()}")

print("Qwen2-VL multimodális modell és processzor betöltése...")
# A Qwen2-VL-7B-Instruct vagy a könnyebb 2B verzió közvetlenül kezeli a képet és a szöveget is
model = Qwen2VLForConditionalGeneration.from_pretrained(
    "Qwen/Qwen2.5-VL-7B-Instruct",  # Ha esetleg szűkös a VRAM, írhatod ide a "Qwen/Qwen2.5-VL-3B-Instruct" nevet is!
    torch_dtype=torch.float16 if device == "cuda" else torch.float32,
    device_map="auto"
)
processor = AutoProcessor.from_pretrained("Qwen/Qwen2.5-VL-7B-Instruct")
print("A modell készen áll a látásra és a szöveges/hang alapú idegenvezetésre!")

def helyszin_es_ajanlo(kep, hang_fajl, turist_leirasa):
    try:
        kontextus_szoveg = []

        # Hang feldolgozása, ha van
        if hang_fajl is not None:
            import whisper
            # Whisper betöltése a hanghoz
            asr_model = whisper.load_model("small")
            result = asr_model.transcribe(hang_fajl, language="hu")
            kontextus_szoveg.append(f"A turista hangüzenete: '{result['text']}'")

        # Szöveg feldolgozása, ha van
        if turist_leirasa:
            kontextus_szoveg.append(f"A turista írásos megjegyzése: '{turist_leirasa}'")

        felhasznaloi_megjegyzes = " ".join(kontextus_szoveg) if kontextus_szoveg == kontextus_szoveg else "Nincs további megjegyzés."

        # Ha nincs kép sem és szöveg sem
        if kep is None and not kontextus_szoveg:
            return "Kérlek tölts fel egy képet, vagy adj meg hangot/szöveget!", "Nincs bemenet."

        # Készítünk egy üzenetet a Qwen2-VL számára (beletesszük a képet és a szöveget is)
        content_list = []
        if kep is not None:
            content_list.append({"type": "image", "image": kep})

        prompt_szoveg = f"""Te egy profi, barátságos magyar idegenvezető vagy.
Nézd meg alaposan a képet, és vedd figyelembe a turista megjegyzését: "{felhasznaloi_megjegyzes}".

A feladatod, hogy pontosan és magyarul válaszolj az alábbi 3 pontban:
1. **Helyszín azonosítása:** Melyik országban, városban vagyunk, és pontosan mi látható a képen?
2. **Ismertető:** Írj róla 2-3 mondatos érdekes, tényszerű leírást.
3. **Programajánló:** Ajánlj 3 konkrét, izgalmas látnivalót vagy programot a közelben!"""

        content_list.append({"type": "text", "text": prompt_szoveg})

        messages = [
            {
                "role": "user",
                "content": content_list
            }
        ]

        # Előkészítés a Qwen2-VL számára
        text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        image_inputs, video_inputs = process_vision_info(messages)

        inputs = processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt"
        ).to(device)

        # Generálás
        with torch.no_grad():
            generated_ids = model.generate(**inputs, max_new_tokens=400, temperature=0.4, do_sample=True)

        generated_ids_trimmed = [
            out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        vegso_valasz = processor.batch_decode(
            generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )[0]

        diagnosztika = f"Qwen2-VL multimodális feldolgozás sikeres.\nMegjegyzés: {felhasznaloi_megjegyzes}"
        return vegso_valasz, diagnosztika

    except Exception as e:
        return f"Hiba történt: {str(e)}", f"Részletek: {str(e)}"

# Gradio felület
demo = gr.Interface(
    fn=helyszin_es_ajanlo,
    inputs=[
        gr.Image(type="pil", label="1. Fotó a helyszínről (Ezt látja a Qwen2-VL)"),
        gr.Audio(sources=["microphone", "upload"], type="filepath", label="2. Hangüzenet"),
        gr.Textbox(label="3. Írd le, mit gondolsz róla", placeholder="Pl.: Melyik országban vagyunk?")
    ],
    outputs=[
        gr.Textbox(label="🌍 Qwen2-VL Idegenvezető Válasza", lines=12),
        gr.Textbox(label="🔗 Rendszer Diagnosztika", lines=5)
    ],
    title="📍 Valódi Multimodális Turisztikai Asszisztens (Qwen2-VL)",
    description="Küldj képet – a modell közvetlenül 'látja' azt, felismeri a helyszínt, és programokat ajánl!"
)

if __name__ == "__main__":
    demo.launch(inbrowser=True)
