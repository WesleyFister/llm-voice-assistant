from lingua import Language, LanguageDetectorBuilder
from openvoice import se_extractor
from openvoice.api import ToneColorConverter
from melo.api import TTS
import os
import wave
import torch
# https://github.com/myshell-ai/OpenVoice/issues/306

class textToSpeech():
    def __init__(self, referenceSpeaker):
        languages = [Language.ENGLISH, Language.SPANISH, Language.FRENCH, Language.CHINESE, Language.JAPANESE, Language.KOREAN]
        self.detector = LanguageDetectorBuilder.from_languages(*languages).build()
        #self.detector = LanguageDetectorBuilder.from_all_languages().with_minimum_relative_distance(0).build() # All languages

    def langDetect(self, text):
        language = self.detector.detect_language_of(text)
        print(language, language.iso_code_639_1.name)
        language = language.iso_code_639_1.name
        
        language = "KR" if language == "KO" else language # KO is the abbrievation for the Korean language but it is KO-KR for South Korea and OpenVoice uses that
        language = "JP" if language == "JA" else language
        return language

    def textToSpeech(self, text, file_name, cuda):
        language = self.langDetect(text)
        
        ckpt_converter = 'checkpoints_v2/converter'
        device = "cuda:0" if torch.cuda.is_available() else "cpu"
        output_dir = 'outputs_v2'

        tone_color_converter = ToneColorConverter(f'{ckpt_converter}/config.json', device=device)
        tone_color_converter.load_ckpt(f'{ckpt_converter}/checkpoint.pth')

        os.makedirs(output_dir, exist_ok=True)

        reference_speaker = 'reference-voices/demo_speaker2.mp3' # This is the voice you want to clone
        target_se, audio_name = se_extractor.get_se(reference_speaker, tone_color_converter, vad=False)

        src_path = 'audio-response/tmp.wav'

        # Speed is adjustable
        speed = 1.0
        model = TTS(language=language, device=device)
        speaker_ids = model.hps.data.spk2id

        for speaker_key in speaker_ids.keys():
            speaker_id = speaker_ids[speaker_key]
            speaker_key = speaker_key.lower().replace('_', '-')
            
            try:
                lang, region = speaker_key.split("-")
            
            except ValueError:
                lang = speaker_key
                region = None
            
            if (lang == "en" and region == "us") or lang != "en":
                source_se = torch.load(f'checkpoints_v2/base_speakers/ses/{speaker_key}.pth', map_location=device)
                model.tts_to_file(text, speaker_id, src_path, speed=speed)
                save_path = file_name

                # Run the tone color converter
                encode_message = "@MyShell"
                tone_color_converter.convert(
                    audio_src_path=src_path, 
                    src_se=source_se, 
                    tgt_se=target_se, 
                    output_path=save_path,
                    message=encode_message)
