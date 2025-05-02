import speech_recognition as sr
from googletrans import Translator
from gtts import gTTS
import io
import pygame

# Initialize pygame mixer
pygame.mixer.init()

def speak(text, lang='en'):
    try:
        tts = gTTS(text=text, lang=lang)
        audio_data = io.BytesIO()
        tts.write_to_fp(audio_data)
        audio_data.seek(0)

        pygame.mixer.music.load(audio_data)
        pygame.mixer.music.play()
        while pygame.mixer.music.get_busy():
            continue
    except ValueError:
        print(f"Text-to-speech not supported for language code: {lang}")

def recognize_and_translate(source_lang_code, target_lang_code):
    recognizer = sr.Recognizer()
    translator = Translator()
    with sr.Microphone() as source:
        print("\nListening... Speak now!")
        recognizer.adjust_for_ambient_noise(source)
        try:
            audio = recognizer.listen(source)
            text = recognizer.recognize_google(audio, language=source_lang_code)
            print(f"You said ({source_lang_code}):", text)

            translated = translator.translate(text, src=source_lang_code, dest=target_lang_code)
            print(f"Translated to {translated.dest}: {translated.text}")
            speak(translated.text, lang=target_lang_code)

        except sr.UnknownValueError:
            print("Sorry, I could not understand the audio.")
        except sr.RequestError:
            print("Could not request results. Please check your internet connection.")

if __name__ == "__main__":
    # Language options
    lang_options = {
        "hi": "Hindi",
        "pa": "Punjabi",
        "bn": "Bengali",
        "gu": "Gujarati",
        "ta": "Tamil",
        "te": "Telugu",
        "ur": "Urdu",
        "en": "English",
        "sa": "Sanskrit",
        "es": "Spanish",
        "fr": "French",
        "zh-cn": "Chinese (Simplified)",
        "la": "Latin"
    }

    print("Available languages:")
    for code, name in lang_options.items():
        print(f"{code} - {name}")

    source_lang = input("Enter source language code (e.g., 'en' for English): ").strip().lower()
    target_lang = input("Enter target language code (e.g., 'hi' for Hindi): ").strip().lower()

    if source_lang not in lang_options:
        print("Invalid source language. Defaulting to English.")
        source_lang = 'en'

    if target_lang not in lang_options:
        print("Invalid target language. Defaulting to Hindi.")
        target_lang = 'hi'

    if target_lang == 'la':
        print("⚠️ Note: Latin is supported for translation but not for speech output.")

    print(f"\nTranslation will be from {lang_options[source_lang]} ({source_lang.upper()}) "
          f"to {lang_options[target_lang]} ({target_lang.upper()})")
    print("Press Ctrl+C to stop.\n")

    try:
        while True:
            recognize_and_translate(source_lang, target_lang)
    except KeyboardInterrupt:
        print("\nExited.")
