from piper.voice import PiperVoice
from piper.download import get_voices
from piper.download import ensure_voice_exists
from melo.api import TTS
import wave
import os
import json

class textToSpeech():
    def __init__(self):        
        model_dir = './piper-models/'
        
        self.file = open(model_dir + 'piper-models.json')
        self.json_models = json.load(self.file)
        
        self.models = {}
        self.download_model()

    def download_model(self, text={"language": None}, model_dir='./piper-models/'):
        # Download Piper TTS model if none are found.
        if self.models == {}:
            download = True
        
        # Check if the language is already in memory
        else:
            for language in self.models:
                download = True
                
                if language == text["language"]:
                    download = False
                    print(f"{text['language']} Already in memory")
                    break
                
        if download == True:
            for language, locale in self.json_models["language"].items():
                for locale, model_name in locale.items():
                    for model_name, model_quality in model_name.items():
                        for model_quality, model_info in model_quality.items():
                            if model_info["enabled"] == True:
                                load_model = False

                                # Load all models with auto_start enabled
                                if text["language"] == None and model_info["auto_start"] == True:
                                    load_model = True

                                # Load the specified language
                                elif text["language"] != None and text["language"] == language:
                                    load_model = True
                                
                                if load_model == True:
                                    # Download model if it doesn't exist
                                    if not os.path.exists(model_dir + model_info["model"] + '.onnx') or not os.path.exists(model_dir + model_info["model"] + '.onnx.json'):
                                        try:
                                            voices_info = get_voices(model_dir, True)
                                            ensure_voice_exists(model_info["model"], model_dir, model_dir, voices_info)
                                            
                                        except:
                                            print("Unable to download model")
                                    
                                    if os.path.exists(model_dir + model_info["model"] + '.onnx') and os.path.exists(model_dir + model_info["model"] + '.onnx.json'):
                                        print(f'Loading {model_info["model"]} TTS model')
                                        self.models[language] = PiperVoice.load(model_dir + model_info["model"] + '.onnx')

                                        # Enable auto start for the loaded model
                                        if model_info["auto_start"] == False:
                                            model_info["auto_start"] = True
                                            json_object = json.dumps(self.json_models, indent=2)
                                            with open("./piper-models/piper-models.json", "w") as file:
                                                file.write(json_object)

    def textToSpeech(self, text, file_name, cuda):    
        self.download_model(text)
        
        for language in self.json_models["language"]:
            if language == text["language"]:
                languageFound = True
                break

            else:
                languageFound = False

        if languageFound == True:
            for locale, model_name in self.json_models["language"][text["language"]].items():
                for model_name, model_quality in model_name.items():
                    for model_quality, model_info in model_quality.items():
                        if model_info["enabled"] == True:
                            wav_file = wave.open(file_name, 'w')
                            audio = self.models[text["language"]].synthesize(text["sentence"], wav_file)
            
        else:
            text["language"] = "KR" if text["language"].upper() == "KO" else text["language"].upper() # KO is the abbrievation for the Korean language but it is KO-KR for South Korea and OpenVoice uses that
            text["language"] = "JP" if text["language"].upper() == "JA" else text["language"].upper()
            
            speed = 1.0
            device = 'auto'

            model = TTS(language=text["language"], device=device) # TODO Preload model so it doesn't unload and reload every single time.
            speaker_ids = model.hps.data.spk2id

            model.tts_to_file(text["sentence"], speaker_ids[text["language"].upper()], file_name, speed=speed)
