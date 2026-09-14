import io
import wave
import os
from pathlib import Path
import pygame
from piper import PiperVoice

# Shared TTS instance to avoid repeated initialization
tts_instance = None

def init_tts(language="zh"):
    """Initialize the TTS engine (once globally)"""
    global tts_instance
    if tts_instance is not None:
        return tts_instance
    
    model_root = Path(os.environ.get("AUTOPATROL_TTS_DIR", Path.home() / "MODELS" / "tts")).expanduser()
    try:
        # Load locally installed Piper voice models.
        if language == 'zh':
            tts_model = str(model_root / "zh" / "zh_CN-huayan-medium.onnx")
            tts_json = tts_model + ".json"
        else:
            tts_model = str(model_root / "en" / "en_US-libritts-high.onnx")
            tts_json = tts_model + ".json"
        
        # Initialize Piper TTS
        synthesizer = PiperVoice.load(tts_model, tts_json)
        synthesizer.volume = 2.0  # Volume
        synthesizer.length_scale = 1.0  # Speech rate
        tts_instance = synthesizer
        print("Piper TTS initialized successfully")
        return synthesizer
    except Exception as e:
        print(f"Piper TTS initialization failed: {e}")
        return None

def synthesize_and_play(text):
    """
    Core interface: synthesize and play text directly (without saving a file)
    :param text: Text to be spoken
    """
    # 1. Initialize TTS (automatically on the first call)
    synthesizer = init_tts()
    if synthesizer is None:
        print(f"TTS not initialized; skipping speech: {text}")
        return
    
    # 2. Synthesize audio into an in-memory byte stream
    try:
        audio_bytes = io.BytesIO()
        with wave.open(audio_bytes, "wb") as wav_file:
            wav_file.setnchannels(1)    # Mono
            wav_file.setsampwidth(2)    # 16 bit samples
            wav_file.setframerate(22050)# Sample rate
            wav_file.setcomptype('NONE', 'NONE')
            synthesizer.synthesize_wav(text, wav_file)
        audio_bytes.seek(0)  # Reset the stream position
    except Exception as e:
        print(f"Audio synthesis failed: {e}; text: {text}")
        return
    
    # 3. Play audio from memory
    try:
        pygame.mixer.init()
        pygame.mixer.music.load(audio_bytes)
        pygame.mixer.music.play()
        # Wait for playback to finish
        while pygame.mixer.music.get_busy():
            pygame.time.Clock().tick(10)
    except Exception as e:
        print(f"Audio playback failed: {e}")
    finally:
        pygame.mixer.quit()

# Test code
if __name__ == "__main__":
    synthesize_and_play("Initializing position")
    synthesize_and_play("Target reached 1.00, 2.00")
