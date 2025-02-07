from piper.voice import PiperVoice
from piper.download import get_voices
from piper.download import ensure_voice_exists
import wave
import os
import json

class textToSpeech():
    def __init__(self, cuda):        
        model_dir = './piper-models/'
        
        self.file = open(model_dir + 'piper-models.json')
        self.json_models = json.load(self.file)
        
        if cuda == True:
            print("Running Piper TTS on GPU inferencing")

        else:
            print("Running Piper TTS on CPU inferencing")

        self.piperModels = {}
        self.download_model(cuda=cuda)

    def download_model(self, text={"language": None}, model_dir='./piper-models/', cuda=False):
        # Download Piper TTS model if none are found.
        if self.piperModels == {}:
            download = True
        
        # Check if the language is already in memory
        else:
            for language in self.piperModels:
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
                                        self.piperModels[language] = PiperVoice.load(model_dir + model_info["model"] + '.onnx', use_cuda=cuda)

                                        # Enable auto start for the loaded model
                                        if model_info["auto_start"] == False:
                                            model_info["auto_start"] = True
                                            json_object = json.dumps(self.json_models, indent=2)
                                            with open("./piper-models/piper-models.json", "w") as file:
                                                file.write(json_object)
                                        
                                        # Piper-TTS with CUDA is faster but only after the first run.
                                        if cuda == True:
                                            wav_file = wave.open("piper-tts-workaround", 'w')
                                            self.piperModels[language].synthesize("Workaround.", wav_file)
                                            os.remove("piper-tts-workaround")

    def cleanText(self, text):
        bannedCharacters = "*$"
        for character in bannedCharacters:
            text["sentence"] = text["sentence"].replace(character, "")

        return text

    def textToSpeech(self, text, file_name, cuda):    
        self.download_model(text=text, cuda=cuda)

        text = self.cleanText(text)
        
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
                            audio = self.piperModels[text["language"]].synthesize(text["sentence"], wav_file)