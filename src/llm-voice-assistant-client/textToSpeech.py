import wave
import os
import json
import requests
from openai import OpenAI
from pathlib import Path
import time

class textToSpeech:
    def __init__(self, base_url, api_key):        
        model_dir = './tts-models/'
        
        self.openai = OpenAI(base_url=base_url, api_key=api_key)
        self.base_url = base_url

        self.file = open(model_dir + 'piper-models.json')
        self.piper_json_models = json.load(self.file)
        self.file = open(model_dir + 'kokoro-models.json')
        self.kokoro_json_models = json.load(self.file)

        self.piperModels = {}
        self.download_model()

    def download_model(self, text={"language": None}, model_dir='./tts-models/'):
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
            for language, locale in self.piper_json_models["language"].items():
                for locale, model_name in locale.items():
                    for model_name, model_quality in model_name.items():
                        for model_quality, model_info in model_quality.items():
                            if model_info["enabled"] == True:
                                if text["language"] != None and text["language"] == language:
                                    # Download model if it doesn't exist
                                    response = requests.post(f"{self.base_url}/models/speaches-ai/piper-{model_info['model']}")

                                    # Enable auto start for the loaded model
                                    if model_info["auto_start"] == False:
                                        model_info["auto_start"] = True
                                        json_object = json.dumps(self.piper_json_models, indent=2)
                                        with open("./tts-models/piper-models.json", "w") as file:
                                            file.write(json_object)

                                # Load all models with auto_start enabled
                                elif text["language"] == None and model_info["auto_start"] == True:
                                    # Download model if it doesn't exist
                                    response = requests.post(f"{self.base_url}/models/speaches-ai/piper-{model_info['model']}")
                                    print(f'Loading {model_info["model"]} TTS model')
                                    res = self.openai.audio.speech.create(model=f"speaches-ai/piper-{model_info['model']}", voice="voice_id", input="Hello world!", response_format="wav")

    def cleanText(self, text):
        bannedCharacters = ["*", "$", "\n"]
        for character in bannedCharacters:
            text["sentence"] = text["sentence"].replace(character, "")

        return text

    def textToSpeech(self, text, file_name):
        text = self.cleanText(text)
        
        for language in self.piper_json_models["language"]:
            if language == text["language"]:
                languageFoundPiper = True
                break

            else:
                languageFoundPiper = False
        
        if languageFoundPiper == False:
            for language in self.kokoro_json_models["language"]:
                if language == text["language"]:
                    languageFoundKokoro = True
                    break

                else:
                    languageFoundKokoro = False

        if languageFoundPiper == True:
            for locale, model_name in self.piper_json_models["language"][text["language"]].items():
                for model_name, model_quality in model_name.items():
                    for model_quality, model_info in model_quality.items():
                        if model_info["enabled"] == True:
                            try:
                                # Generate audio
                                res = self.openai.audio.speech.create(model=f"speaches-ai/piper-{model_info['model']}", voice=f"speaches-ai/piper-{model_info['model']}", input=text["sentence"], response_format="wav")
                                with Path(file_name).open("wb") as f:
                                    f.write(res.response.read())
                            
                            except Exception as e:
                                # Check if the error is due to the model not being installed locally
                                if e.response.status_code == 404 and "not installed locally" in str(e):
                                    # Download the model
                                    self.download_model(text=text)

                                    # Generate audio
                                    res = self.openai.audio.speech.create(model=f"speaches-ai/piper-{model_info['model']}", voice=f"speaches-ai/piper-{model_info['model']}", input=text["sentence"], response_format="wav")
                                    with Path(file_name).open("wb") as f:
                                        f.write(res.response.read())
                                    
                                else:
                                    raise

        elif languageFoundKokoro == True:
            for locale, model_name in self.kokoro_json_models["language"][text["language"]].items():
                for model_name, model_info in model_name.items():
                    if model_info["enabled"] == True:
                        res = self.openai.audio.speech.create(model="speaches-ai/Kokoro-82M-v1.0-ONNX", voice=model_info['model'], input=text["sentence"], response_format="wav")
                        with Path(file_name).open("wb") as f:
                            f.write(res.response.read())