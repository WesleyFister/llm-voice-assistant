from piper.voice import PiperVoice
from piper.download import get_voices
from piper.download import ensure_voice_exists
import wave
import os

class textToSpeech():
    def __init__(self, tts_model):
        # Download Piper TTS model if not found.
        model_dir = './piper-models/'
        if os.path.exists(model_dir + tts_model + '.onnx') != True or os.path.exists(model_dir + tts_model + '.onnx') != True:
            voices_info = get_voices(model_dir, False)
            ensure_voice_exists(tts_model, model_dir, model_dir, voices_info)
        
        self.voice = PiperVoice.load(model_dir + tts_model + '.onnx')

    def textToSpeech(self, text, file_name, cuda):    
        wav_file = wave.open(file_name, 'w')
        audio = self.voice.synthesize(text, wav_file)