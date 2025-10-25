from openai import OpenAI
from lingua import Language, LanguageDetectorBuilder
from nltk.tokenize import sent_tokenize
import nltk
from datetime import datetime
import os
import json
import requests

class textToText:
    def __init__(self, llm_model, base_url, llm_api_key):
        # Supported languages by TTS
        languagesPiperTTS = [Language.ARABIC, Language.CATALAN, Language.CZECH, Language.WELSH, Language.DANISH, Language.GERMAN, Language.GREEK, Language.ENGLISH, Language.SPANISH, Language.PERSIAN, Language.FINNISH, Language.FRENCH, Language.HUNGARIAN, Language.ICELANDIC, Language.ITALIAN, Language.GEORGIAN, Language.KAZAKH, Language.DUTCH, Language.POLISH, Language.PORTUGUESE, Language.ROMANIAN, Language.RUSSIAN, Language.SLOVAK, Language.SERBIAN, Language.SWEDISH, Language.SWAHILI, Language.TURKISH, Language.UKRAINIAN, Language.VIETNAMESE, Language.CHINESE]
        languagesKokoroTTS = [Language.CHINESE, Language.ENGLISH, Language.FRENCH, Language.HINDI, Language.ITALIAN, Language.JAPANESE, Language.PORTUGUESE, Language.SPANISH]
        languages = list(set(languagesPiperTTS + languagesKokoroTTS))

        self.detector = LanguageDetectorBuilder.from_languages(*languages).with_preloaded_language_models().with_minimum_relative_distance(0.3).build() # Eager load language detection models.
        
        self.llm_model = llm_model
        
        # If Ollama is installed automatically download LLM.
        ollama_installed = self.ollamaDownloadModel(base_url, llm_model)

        if ollama_installed == True:
            self.client = OpenAI(
                base_url = base_url,
                api_key = llm_api_key, # Required even if unused.
            )

        else:
            self.client = OpenAI(
                base_url = base_url,
                api_key = llm_api_key, # Required even if unused.
            )

        nltk.download('punkt_tab')

        # Without this the first response from the LLM is very slow. Getting a response first speeds it up but only by running this complete function first fully reduces the delay. I am not sure as why.
        messages = self.chatWithHistory({"language": "en", "transcript": "Copy me verbatim 'The quick brown fox jumps over the lazy dog.'"}, "chat-history/workAround.json", "Follow the user's instructions.")
        for message in messages:
            message = 0
        os.remove("chat-history/workAround.json")

    def ollamaDownloadModel(self, base_url, llm_model):
        try:
            ollama_url = f"{base_url}/api/show"

            response = requests.post(ollama_url, data=json.dumps({"model": llm_model}), stream=False)

            if response.status_code == 404:
                ollama_url = f"{base_url}/api/pull"

                response = requests.post(ollama_url, data=json.dumps({"model": llm_model}), stream=True)

                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:
                        chunk_data = json.loads(chunk.decode('utf-8'))
                        completed = chunk_data.get("completed")
                        total = chunk_data.get("total")
                        if completed is not None:
                            percentage = round((completed / total) * 100, 2)
                            print(f"\x1b[2KDownloading {llm_model}: {percentage}%", end="\r")

            return True

        except requests.exceptions.RequestException:
            print("Ollama isn't installed so model can't be downloaded automatically")

            return False

    def langDetect(self, text, transcription):
        language = self.detector.detect_language_of(text)
        
        if language != None:
            print(language, language.iso_code_639_1.name)
            language = language.iso_code_639_1.name
        
        language = transcription["language"] if language == None else language
        return language.lower()

    def chatWithHistory(self, transcription, chatHistoryFile, systemPrompt=""):        
        # Add date and time into to system prompt
        systemPrompt = systemPrompt.format(
            date=datetime.now().strftime("%Y-%m-%d (%A)"),
            time=datetime.now().strftime("%I:%M %p")
        )
        
        # Try to load the chat history from a file
        if os.path.exists(chatHistoryFile):
            with open(chatHistoryFile, "r") as file:
                json_string = file.read()

            chatHistory = json.loads(json_string)

        # If the file doesn't exist, create it with an initial system prompt
        else:
            json_string = json.dumps(
                [
                    {
                    'role': 'system',
                    'content': systemPrompt,
                    }
                ]
            )

            with open(chatHistoryFile, 'w') as file:
                file.write(json_string)
                
            with open(chatHistoryFile, "r") as file:
                json_string = file.read()

            chatHistory = json.loads(json_string)

        # Append the user's message to the chat history
        chatHistory.append(
            {
            'role': 'user',
            'content': f'<context>Current date: {datetime.now().strftime("%Y-%m-%d (%A)")}\nCurrent time: {datetime.now().strftime("%I:%M %p")}]</context>\n\n {transcription["transcript"]}',
            }
        )

        stream = self.client.chat.completions.create(
            model = self.llm_model, 
            messages = chatHistory,
            stream = True,
        )

        sentence = 1
        response = ""
        for chunk in stream:
            if chunk.choices[0].delta.content != None:
                part = chunk.choices[0].delta.content
                response = response + part
            
                # Chunk the response into sentences and yield each one as it is completed
                sentences = sent_tokenize(response)
                while sentence < len(sentences):
                    language = self.langDetect(sentences[sentence - 1], transcription)
                    yield { "token": part, "sentence": sentences[sentence - 1], "language": language }
                    sentence += 1
                
                else:
                    yield { "token": part, "sentence": "" }

        language = self.langDetect(sentences[sentence - 1], transcription)
        yield { "token": part, "sentence": sentences[sentence - 1], "language": language }

        # Append the assistant's response to the chat history
        chatHistory.append(
            {
            'role': 'assistant',
            'content': response,
            }
        )

        # Delete the info role so it doesn't hog the LLM's context window
        chatHistory[-2] = {
            'role': 'user',
            'content': transcription["transcript"],
            }

        # Write chat history to file
        json_string = json.dumps(chatHistory, indent=2)
        with open(chatHistoryFile, 'w') as file:
            file.write(json_string)