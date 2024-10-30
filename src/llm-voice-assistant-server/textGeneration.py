from datetime import datetime
import json
import ollama
import nltk
from nltk.tokenize import sent_tokenize
from lingua import Language, LanguageDetectorBuilder

class ollamaChat:
    def __init__(self, llm_model):
        # Supported languages by TTS
        languagesPiperTTS = [Language.ARABIC, Language.CATALAN, Language.CZECH, Language.WELSH, Language.DANISH, Language.GERMAN, Language.GREEK, Language.ENGLISH, Language.SPANISH, Language.PERSIAN, Language.FINNISH, Language.FRENCH, Language.HUNGARIAN, Language.ICELANDIC, Language.ITALIAN, Language.GEORGIAN, Language.KAZAKH, Language.DUTCH, Language.POLISH, Language.PORTUGUESE, Language.ROMANIAN, Language.RUSSIAN, Language.SLOVAK, Language.SERBIAN, Language.SWEDISH, Language.SWAHILI, Language.TURKISH, Language.UKRAINIAN, Language.VIETNAMESE, Language.CHINESE]
        languagesMeloTTS = [Language.ENGLISH, Language.SPANISH, Language.FRENCH, Language.CHINESE, Language.JAPANESE, Language.KOREAN]
        languages = list(set(languagesPiperTTS + languagesMeloTTS))
        self.detector = LanguageDetectorBuilder.from_languages(*languages).with_minimum_relative_distance(0.3).build()
        #self.detector = LanguageDetectorBuilder.from_all_languages().with_minimum_relative_distance(0).build() # All languages
        
        self.llm_model = llm_model

        # Check if the model is available, and download it if not
        try:
            ollama.show(self.llm_model)
            
        except ollama._types.ResponseError:
            print(f'Downloading {self.llm_model} LLM model')
            ollama.pull(self.llm_model)

        print(f"Loading {self.llm_model}")
        ollama.chat(model=self.llm_model)
        
        nltk.download('punkt_tab')

    def langDetect(self, text, transcription):
        language = self.detector.detect_language_of(text)
        
        if language != None:
            print(language, language.iso_code_639_1.name)
            language = language.iso_code_639_1.name
        
        language = transcription["language"] if language == None else language
        return language.lower()

    def chatWithHistory(self, transcription, chatHistoryFile, systemPrompt):
        systemPrompt = f'{systemPrompt} Current date: {datetime.now().strftime("%Y-%m-%d (%A)")} Current time: {datetime.now().strftime("%I:%M %p")}'
        
        # Try to load the chat history from a file
        try:
            with open(chatHistoryFile, "r") as file:
                json_string = file.read()

            chatHistory = json.loads(json_string)

        # If the file doesn't exist, create it with an initial system prompt
        except FileNotFoundError:
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
            'content': transcription["transcript"],
            }
        )

        stream = ollama.chat(
            model=self.llm_model, 
            messages=chatHistory,
            stream=True,
        )

        sentence = 1
        response = ""
        for chunk in stream:
            part = chunk['message']['content']
            response = response + part
            
            # Chunk the response into sentences and yield each one as it is completed
            sentences = sent_tokenize(response)
            if len(sentences) > sentence:
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

        # Write chat history to file
        json_string = json.dumps(chatHistory)
        with open(chatHistoryFile, 'w') as file:
            file.write(json_string)
