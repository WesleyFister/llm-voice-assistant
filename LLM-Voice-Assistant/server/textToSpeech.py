from piper.voice import PiperVoice
from piper.download import get_voices
from piper.download import ensure_voice_exists
import wave
import os

def textToSpeech(tts_model, text, fileName):
    modelDir = './piper-models/'
    
    # Download Piper TTS model if not found.
    if os.path.exists(modelDir + tts_model + '.onnx') != True or os.path.exists(modelDir + tts_model + '.onnx') != True:
        voices_info = get_voices(modelDir, False)
        ensure_voice_exists(tts_model, modelDir, modelDir, voices_info)

    voice = PiperVoice.load(modelDir + tts_model + '.onnx')
    wav_file = wave.open(fileName, 'w')
    audio = voice.synthesize(text, wav_file)
