from wakeWord import wakeWord
from textToText import textToText
from textToSpeech import textToSpeech
import os
import time
import tomllib
import threading
import socket
import wave
import queue
import numpy as np
import pyaudio
import simpleaudio as sa
import torch
torch.set_num_threads(1)
import torchaudio
from collections import deque
from pathlib import Path
from openai import OpenAI
# possibly can remove torch by using silero vad onnx

class llmVoiceAssistantClient():
    def __init__(self):
        os.makedirs("audio-input", exist_ok=True)
        os.makedirs("audio-response", exist_ok=True)
        os.makedirs("chat-history", exist_ok=True)
        
        with open("config.toml", 'rb') as f:
            config = tomllib.load(f)
        
        # num_samples / SAMPLE_RATE = seconds of audio
        # In this case it is 0.032 seconds.
        mult = 31.25
        self.recording_length = int(30 * mult)
        self.vad_initial_delay = int(config['vad']['initial_delay'] * mult)
        self.vad_delay = int(config['vad']['delay'] * mult)
        self.no_wakeword = config['system']['disable_wakeword']
        if config['chat']['history'] == "":
            self.chat_history = f"chat-history/chat-history-{os.urandom(8).hex()}.json"

        else:
            self.chat_history = config['chat']['history']

        prompt_file = config['system']['system_prompt']
        with open(prompt_file, 'r') as f:
            self.system_prompt = f.read()

        self.stt_model = config['stt']['model']
        self.stt_language = config['stt']['language']
        self.client_transcribe = OpenAI(base_url=config['stt']['api'], api_key=config['stt']['api_key'])
        self.textToText = textToText(base_url=config['llm']['api'], llm_api_key=config['llm']['api_key'], llm_model=config['llm']['model'])
        self.textToSpeech = textToSpeech(base_url=config['tts']['api'], api_key=config['tts']['api_key'])

        # Have whisper transcribe something to preload it into memory
        with Path("workAround.wav").open("rb") as audio_file:
            self.client_transcribe.audio.transcriptions.create(model=self.stt_model, language=self.stt_language, response_format="verbose_json", file=audio_file)

    def sendTextResponseToClient(self, transcription, sentences, done):
        sentinel = 0
        response = ""
        startTime = time.perf_counter()
        messages = self.textToText.chatWithHistory(transcription, self.chat_history, self.system_prompt)
        #print(f"\033[34mAI: \033[0m", end='', flush=True)
        for message in messages:
            if self.running == True:
                #print(f"\033[34m{message['token']}\033[0m", end='', flush=True)
                
                if message["sentence"] != "":
                    print(f"\033[34mAI: {message['sentence']}\033[0m")

                    if sentinel == 0:
                        endTime = time.perf_counter()
                        print(f"Took {endTime - startTime:.4f} seconds LLM response")
                    
                    sentinel = 1
                    
                    sentences.put(message)
        
        print("Finished generating output")
        sentences.put(done)

    def clientStart(self):
        while True:
            if self.no_wakeword == False:
                wakeWord()

                self.running = False

            self.transcription = self.recordAudio()
            self.running = True
            
            if self.no_wakeword == False:
                getAudio_thread = threading.Thread(target=self.playAudioResponse)
                getAudio_thread.start()
            
            else:
                self.playAudioResponse()

    def recordAudio(self):
        # Download and load the Silero VAD model.
        model, utils = torch.hub.load(repo_or_dir='snakers4/silero-vad', model='silero_vad')
        torch.set_grad_enabled(False) # Not having this causes memory leak in SileroVAD as audio will be stored indefinitely.

        # Provided by Alexander Veysov.
        def int2float(sound):
            abs_max = np.abs(sound).max()
            sound = sound.astype('float32')
            if abs_max > 0:
                sound *= 1/32768
                sound = sound.squeeze()  # depends on the use case.
                return sound

        print("Started Recording")
        if self.no_wakeword == False:
            # Load and play an audio file.
            audioFile = sa.WaveObject.from_wave_file("media/on.wav")
            play_obj = audioFile.play()

        p = pyaudio.PyAudio()
        FORMAT = pyaudio.paInt16
        CHANNELS = 1
        SAMPLE_RATE = 16000
        CHUNK = int(SAMPLE_RATE / 10)
        num_samples = 512

        audioInput = 'audio-input/input-' + os.urandom(8).hex() + '.wav' 
        wf = wave.open(audioInput, 'wb')
        wf.setnchannels(CHANNELS)
        wf.setsampwidth(p.get_sample_size(FORMAT))
        wf.setframerate(SAMPLE_RATE)

        stream = p.open(format=FORMAT, channels=CHANNELS, rate=SAMPLE_RATE, input=True, frames_per_buffer=CHUNK)

        silence = 0
        voiceDetected = False
        delay = self.vad_initial_delay
        delay2 = self.recording_length
        new_confidence = 1
        buffer_written = False
        buffer_size = 93
        audio_buffer = deque(maxlen=buffer_size)

        # Send user audio data to the server.
        while silence < delay or voiceDetected == False:
            if self.no_wakeword == True:
                print(f"\x1b[2K{silence}", end="\r")

            else:
                print(f"\x1b[2K{silence}/{delay}", end="\r") # \x1b[2K ANSI sequence for clearing the current line.

            silence += 1
            audio_chunk = stream.read(num_samples)

            audio_buffer.append(audio_chunk)
            if voiceDetected == True:
                if buffer_written == False:
                    buffer_written = True
                    for chunk in audio_buffer:
                        wf.writeframes(chunk)

                else:
                    wf.writeframes(audio_chunk)

            audio_int16 = np.frombuffer(audio_chunk, np.int16)

            # Check if amplitude is 0. This is necessary because if amplitude = 0 then it will crash the program. This needs to be fixed because it makes the VAD delay inconsistent.
            amplitude = np.max(np.abs(audio_int16))
            if amplitude == 0:
                continue
            
            audio_float32 = int2float(audio_int16)

            # Get the confidences
            new_confidence = model(torch.from_numpy(audio_float32), 16000).item()
            if new_confidence >= 0.3:
                delay = self.vad_delay
                delay2 = self.vad_delay
                silence = 0
                voiceDetected = True

        stream.stop_stream()
        stream.close
        wf.close()

        print(f'{silence}/{delay}')
        print("Stopped recording")
        if self.no_wakeword == False:
            # Load and play an audio file.
            audioFile = sa.WaveObject.from_wave_file("media/off.wav")
            play_obj = audioFile.play()

        if voiceDetected == False:
            print("Voice was not detected")
            self.playAudioResponse()
            self.clientStart()

        if voiceDetected == True:
            with Path(audioInput).open("rb") as audio_file:
                transcription = {}

                startTime = time.perf_counter()
                transcription_json = self.client_transcribe.audio.transcriptions.create(model=self.stt_model, language=self.stt_language, response_format="verbose_json", file=audio_file)
                endTime = time.perf_counter()
                print(f"Took {endTime - startTime:.4f} seconds to transcribe User\'s speech")

                transcription["transcript"] = transcription_json.text
                transcription["language"] = transcription_json.language
                print(f"\033[32mUser: {transcription['transcript']}\033[0m")

            if os.path.exists(audioInput):
                os.remove(audioInput)

        return transcription

    def playAudioResponse(self):
        def getResponse(q):
            self.sendTextResponseToClient(self.transcription, sentences, done)
        
        def getAudio(sentences, audioFile):
            while self.running == True:
                sentence = sentences.get()

                if sentence != done:
                    audioResponse = 'audio-response/response-' + os.urandom(8).hex() + '.wav'
                    
                    startTime = time.perf_counter()
                    self.textToSpeech.textToSpeech(sentence, audioResponse)
                    endTime = time.perf_counter()
                    print(f"Took {endTime - startTime:.4f} seconds to generate audio")

                    audioFile.put(audioResponse)
                
                else:
                    audioFile.put(done)
                    print("Finished generating audio")
                    break

        def playAudio(audioFile):
            while True:
                audioResponse = audioFile.get()

                if audioResponse != done:
                    if self.running == True:
                        saAudioFile = sa.WaveObject.from_wave_file(audioResponse)
                        play_obj = saAudioFile.play()
                        play_obj.wait_done() # Makes program wait for audio to finish.

                    if os.path.exists(audioResponse):
                        os.remove(audioResponse)

                else:
                    print("Finished playing audio")
                    break

        done = os.urandom(8).hex()

        # Create a queue
        sentences = queue.Queue()
        audioFile = queue.Queue()
        
        # Create and start the producer thread
        getResponse_thread = threading.Thread(target=getResponse, args=(sentences,))
        getResponse_thread.start()

        # Create and start the producer thread
        getAudio_thread = threading.Thread(target=getAudio, args=(sentences, audioFile,))
        getAudio_thread.start()

        # Create and start the consumer thread
        playAudio_thread = threading.Thread(target=playAudio, args=(audioFile,))
        playAudio_thread.start()

        # Wait for the producer to finish
        getResponse_thread.join()

        # Wait for the producer to finish
        getAudio_thread.join()

        # Wait for the consumer to finish
        playAudio_thread.join()

if __name__ == '__main__':
    client = llmVoiceAssistantClient()
    client.clientStart()