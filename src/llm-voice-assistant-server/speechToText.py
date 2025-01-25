from faster_whisper import WhisperModel
from faster_whisper.utils import download_model
from scipy.io import wavfile
import time

class speechToText:
    def __init__(self, stt_model, cuda):
        self.stt_model = stt_model
        
        download_model(self.stt_model)

        if cuda == True:
            # or run on GPU with INT8
            print('Running Whisper on GPU inferencing')
            self.model = WhisperModel(self.stt_model, device="cuda", compute_type="int8")

        else:
            # or run on CPU with INT8
            print('Running Whisper on CPU inferencing')
            self.model = WhisperModel(self.stt_model, device="cpu", compute_type="int8")

        self.transcribe("workAround.wav") # Despite loading the model beforehand the first transcription always takes longer. So we transcribe a dummy audio file first.

    def detect_language(self, audioInput, allowed_languages):
        sampling_rate, audio_data = wavfile.read(audioInput)

        language, language_probability, all_language_probs = self.model.detect_language(audio_data)

        score = 0
        for language_code, language_prob in all_language_probs:
            for allowed_language in allowed_languages:
                if language_code == allowed_language:
                    if language_prob > score:
                        score = language_prob
                        detected_language = language_code

        return detected_language

    def transcribe(self, audioInput):
        transcript = ""

        if self.model.model.is_multilingual:
            languagesPiperTTS = ['ar', 'ca', 'cs', 'cy', 'da', 'de', 'el', 'en', 'es', 'fa', 'fi', 'fr', 'hu', 'is', 'it', 'ka', 'kk', 'nl', 'pl', 'pt', 'ro', 'ru', 'sk', 'sr', 'sv', 'sw', 'tr', 'uk', 'vi', 'zh']
            allowed_languages = languagesPiperTTS
            #allowed_languages = list(set(languagesPiperTTS + otherTTSEngines))
            detected_language = self.detect_language(audioInput, allowed_languages)

        else:
            detected_language = 'en'

        segments, info = self.model.transcribe(audioInput, beam_size=5, language=detected_language)
        for segment in segments:
            transcript += segment.text

        return { "transcript": transcript, "language": info.language }