import gradio as gr
import librosa
import torch
from transformers import pipeline

# Eszköz beállítása (ha van Nvidia GPU-d, azt használja, különben a processzort)
device = 0 if torch.cuda.is_available() else -1
print(
    f"Futtatási környezet: {'GPU (CUDA)' if device == 0 else 'CPU (Ez kicsit lassabb lehet)'}"
)

print("1/3 Képmodell betöltése...")
vision_pipe = pipeline(
    "image-classification", model="google/vit-base-patch16-224", device=device
)

print("2/3 Hangmodell betöltése...")
asr_pipe = pipeline(
    "automatic-speech-recognition", model="openai/whisper-small", device=device
)

print("3/3 Nyelvi modell betöltése...")
llm_pipe = pipeline(
    "text-generation", model="Qwen/Qwen2.5-1.5B-Instruct", device_map="auto"
)

print("Minden modell készen áll!")


def helyszin_es_ajanlo(kep, hang_fajl, turist_leirasa):
    try:
        kontextus_darabok = []

        # 1. Kép elemzése
        if kep is not None:
            kep_eredmeny = vision_pipe(kep)
            top_talalat = kep_eredmeny[0]["label"]
            kontextus_darabok.append(
                f"- A vizuális felismerő szerint ez látható a képen: {top_talalat}."
            )

        # 2. Hangüzenet biztonságos feldolgozása librosa-val
        if hang_fajl is not None:
            speech_array, sampling_rate = librosa.load(hang_fajl, sr=16000)
            hang_eredmeny = asr_pipe(
                {"array": speech_array, "sampling_rate": sampling_rate},
                generate_kwargs={"language": "hungarian"},
            )
            hang_szoveg = hang_eredmeny["text"]
            kontextus_darabok.append(
                f"- A turista hangüzenetben ezt mondta: '{hang_szoveg}'"
            )

        # 3. Szöveges leírás
        if turist_leirasa:
            kontextus_darabok.append(
                f"- A turista írásos megjegyzése: '{turist_leirasa}'"
            )

        if not kontextus_darabok:
            return (
                "Kérlek add meg legalább a képet, egy hangot vagy egy leírást!",
                "Nincs bemenet.",
            )

        osszegyujtott_adatok = "\n".join(kontextus_darabok)

        # Chat formátum a Qwen modell számára szigorúbb, pontosabb instrukciókkal
        uzenetek = [
            {
                "role": "system",
                "content": "Te egy profi, barátságos magyar idegenvezető vagy. Pontos, logikus válaszokat adsz, kerülsz minden halandzsát.",
            },
            {
                "role": "user",
                "content": f"Íme a turista által küldött adatok:\n{osszegyujtott_adatok}\n\nA feladatod, hogy pontosan és magyarul válaszolj az alábbi 3 pontban:\n1. Helyszín azonosítása: Nevezd meg pontosan az épületet vagy nevezetességet.\n2. Ismertető: Írj róla 2-3 mondatos érdekes leírást.\n3. Programajánló: Ajánlj 3 látnivalót a közelben!",
            },
        ]
        text = llm_pipe.tokenizer.apply_chat_template(
            uzenetek, tokenize=False, add_generation_prompt=True
        )
        # Generálás beállítások a pontosabb válaszért
        valasz = llm_pipe(
            text,
            max_new_tokens=350,
            do_sample=True,
            temperature=0.3,
            return_full_text=False,
            pad_token_id=llm_pipe.tokenizer.eos_token_id
        )
        vegso_valasz = valasz[0]["generated_text"]

        diagnosztika = f"Feldolgozott adatok:\n{osszegyujtott_adatok}"
        return vegso_valasz, diagnosztika

    except Exception as e:
        return f"Hiba történt: {str(e)}", f"Részletek: {str(e)}"


# Gradio felület
demo = gr.Interface(
    fn=helyszin_es_ajanlo,
    inputs=[
        gr.Image(type="pil", label="1. Fotó a helyszínről"),
        gr.Audio(
            sources=["microphone", "upload"],
            type="filepath",
            label="2. Hangüzenet",
        ),
        gr.Textbox(
            label="3. Írd le, mit látlsz vagy mit tudsz róla",
            placeholder="Pl.: Hatalmas ókori aréna Rómában...",
        ),
    ],
    outputs=[
        gr.Textbox(label="🌍 Helyszín azonosítása és Látnivaló Ajánló", lines=12),
        gr.Textbox(label="🔗 Rendszer Diagnosztika", lines=5),
    ],
    title="📍 Multimodális Turisztikai Asszisztens",
    description=(
        "Küldj képet, hangot vagy leírást – a rendszer megmondja hol vagy és"
        " programokat ajánl!"
    ),
)

if __name__ == "__main__":
    demo.launch(inbrowser=True)
