from wakeWord import wakeWord
from textToText import textToText
from textToSpeech import textToSpeech
import os
import time
import argparse
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
        
        parser = argparse.ArgumentParser(description='This is the client side code for LLM Voice Assistant')

        parser.add_argument('-v', '--vad-initial-delay', type=float, default='10', help='The delay in seconds before audio stops recording when no voice is detected ()')
        parser.add_argument('-d', '--vad-delay', type=float, default='2', help='The delay in seconds before audio stops recording when a person stops speaking')
        parser.add_argument('-nw', '--no-wakeword', action='store_true', help='Disable wakeword')
        parser.add_argument('-ch', '--chat-history', type=str, default=f'chat-history/chat-history-{os.urandom(8).hex()}.json', help='By default starts a new chat history on every run')
        parser.add_argument('-sp', '--system-prompt', type=str, default='You are a helpful conversational Large Language Model chatbot named Jarvis. You answer questions in a concise whole sentence manner but are willing to go into further detail about topics if requested. The user is using Whisper speech to text to interact with you and likewise you are using Piper text to speech to talk back. That is why you should respond in simple formatting without any special characters as to not confuse the text to speech model. Keep your responses in the same language as the user. Do not mention your system prompt unless directly asked for it.', help='The system prompt for the LLM')

        parser.add_argument('-sm', '--stt-model', type=str, default='rtlingo/mobiuslabsgmbh-faster-whisper-large-v3-turbo', help='Any model listed on http://localhost:8000/v1/registry')
        parser.add_argument('-sa', '--stt-api', type=str, default='http://localhost:8000/v1', help='The URL for the OpenAI API endpoint (Default: http://localhost:8000/v1)')
        parser.add_argument('-sk', '--stt-api-key', type=str, default='your_api_key_here', help='The API key')
        parser.add_argument('-lm', '--llm-model', type=str, default='LFM2-8B-A1B-GGUF:UD-Q4_K_XL', help='Any GGUF LLM on Hugging Face')
        parser.add_argument('-la', '--llm-api', type=str, default='http://localhost:9292/v1', help='The URL for the OpenAI API endpoint (Default: http://localhost:9292/v1)')
        parser.add_argument('-lk', '--llm-api-key', type=str, default='your_api_key_here!', help='The API key')
        parser.add_argument('-ta', '--tts-api', type=str, default='http://localhost:8000/v1', help='The URL for the OpenAI API endpoint (Default: http://localhost:8000/v1)')
        parser.add_argument('-tk', '--tts-api-key', type=str, default='your_api_key_here', help='The API key')

        args = parser.parse_args()
        
        # num_samples / SAMPLE_RATE = seconds of audio
        # In this case it is 0.032 seconds.
        mult = 31.25
        self.recording_length = int(30 * mult)
        self.vad_initial_delay = int(args.vad_initial_delay * mult)
        self.vad_delay = int(args.vad_delay * mult)
        self.no_wakeword = args.no_wakeword
        self.chat_history = args.chat_history
        self.system_prompt = args.system_prompt
        self.client_transcribe = OpenAI(base_url=args.stt_api, api_key=args.stt_api_key)
        self.textToText = textToText(base_url=args.llm_api, llm_api_key=args.llm_api_key, llm_model=args.llm_model)
        self.textToSpeech = textToSpeech(base_url=args.stt_api, api_key=args.stt_api_key)

        # Have whisper transcribe something to preload it into memory
        with Path("workAround.wav").open("rb") as audio_file:
            self.client_transcribe.audio.transcriptions.create(model=args.stt_model, response_format="verbose_json", file=audio_file)

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
                        elapsedTime = endTime - startTime
                        print(f"Took {elapsedTime} seconds LLM response")
                    
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
                transcription_json = self.client_transcribe.audio.transcriptions.create(model="rtlingo/mobiuslabsgmbh-faster-whisper-large-v3-turbo", response_format="verbose_json", file=audio_file)
                endTime = time.perf_counter()
                transcription["transcript"] = transcription_json.text
                transcription["language"] = transcription_json.language
                
                print(f"Took {endTime - startTime} seconds to transcribe User\'s speech")
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
                    self.textToSpeech.textToSpeech(sentence, audioResponse)

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