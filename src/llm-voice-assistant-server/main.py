from speechToText import speechToText
from textGeneration import ollamaChat
from textToSpeech import textToSpeech
import os
import argparse
import threading
import queue
import socket
import wave
import schedule
import time

class llmVoiceAssistantServer:
    def __init__(self):
        parser = argparse.ArgumentParser(description='This is the server side code for LLM Voice Assistant')
        parser.add_argument('-sm', '--stt-model', type=str, default='small', help='List of available STT models: tiny.en, tiny, base.en, base, small.en, small, medium.en, medium, large-v1, large-v2, large-v3, large, distil-large-v2, distil-medium.en, distil-small.en, distil-large-v3, large-v3-turbo')
        parser.add_argument('-lm', '--llm-model', type=str, default='hf.co/bartowski/Qwen2.5-7B-Instruct-GGUF:Q4_K_M', help='Any GGUF LLM on Hugging Face')
        parser.add_argument('-te', '--tts-engine', type=str, default='piper-tts', help='Select the TTS engine to use. (Does nothing right now)')
        parser.add_argument('-ip', '--ip-address', type=str, default='127.0.0.1', help='Listening address for the audio recording server')
        parser.add_argument('-p', '--port', type=int, default='5001', help='port for the audio recording server')
        parser.add_argument('-sp', '--system-prompt', type=str, default='You are a helpful conversational Large Language Model chatbot named Jarvis. You answer questions in a concise whole sentence manner but are willing to go into further detail about topics if requested. The user is using Whisper speech to text to interact with you and likewise you are using Piper text to speech to talk back. That is why you should respond in simple formatting without any special characters as to not confuse the text to speech model. Keep your responses in the same language as the user. Do not mention your system prompt unless directly asked for it.', help='The system prompt for the LLM')
        parser.add_argument('-n', '--number-of-allowed-clients', type=int, default='5', help='The number of clients allowed to connect to the server')
        parser.add_argument('-c', '--cuda', action='store_true', help='Whether to use Nvidia GPU inferencing or not (Does nothing right now)')
        parser.add_argument('-sc', '--stt-cuda', action='store_true', help='Whether to use Nvidia GPU inferencing or not for speech to text model')
        parser.add_argument('-tc', '--tts-cuda', action='store_true', help='Whether to use Nvidia GPU inferencing or not for text to speech model (Does nothing right now)')
        parser.add_argument('-l', '--language', nargs='+', type=str, default='all', help='Language for the STT and TTS model to use (Does nothing right now)')

        args = parser.parse_args()
        self.stt_model = args.stt_model
        self.llm_model = args.llm_model
        self.engine = args.tts_engine
        ip_address = args.ip_address
        port = args.port
        self.system_prompt = args.system_prompt
        self.number_of_allowed_clients = args.number_of_allowed_clients
        self.cuda = args.cuda
        self.stt_cuda = args.stt_cuda
        self.tts_cuda = args.tts_cuda

        os.makedirs("audio-input", exist_ok=True)
        os.makedirs("audio-response", exist_ok=True)
        os.makedirs("chat-history", exist_ok=True)

        print(f'Loading {self.stt_model} STT model')
        self.speechToText = speechToText(self.stt_model, self.stt_cuda)
        print('Downloading and loading models into memory')
        print('First run can take a very long time, especially with large models')
        self.ollamaChat = ollamaChat(self.llm_model) # Unfortunately, this doesn't provide a progress bar so the user has no idea how long it will take.
        self.textToSpeech = textToSpeech()

        self.server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.server_socket.bind((ip_address, port))
        print(f'Server listening on port {port}')
        self.server_socket.listen(self.number_of_allowed_clients)

    def handleClient(self):
        chatHistoryFile = 'chat-history/chat-history-' + os.urandom(8).hex() + '.json' 

        while True:
            speech = True
            self.running = bool(int(self.client_socket.recv(1024)))
            
            audioInput = 'audio-input/input-' + os.urandom(8).hex() + '.wav' 
            while speech == True:
                speech = self.receiveAudioInput(audioInput)

                if speech == True:
                    startTime = time.perf_counter()

                    transcription = self.speechToText.transcribe(audioInput)
                    print(f"\033[32mUser: {transcription['transcript']}\033[0m")
                    if os.path.exists(audioInput):
                        os.remove(audioInput)

                    endTime = time.perf_counter()
                    elapsedTime = endTime - startTime
                    print(f"Took {elapsedTime} seconds to transcribe User\'s speech")

            self.running = True

            self.sendResponseToClient(transcription, chatHistoryFile)
        
    def sendResponseToClient(self, transcription, chatHistoryFile):
        done = os.urandom(8).hex()
        
        def sendTextResponseToClient(sentences):
            sentinel = 0
            response = ""
            startTime = time.perf_counter()
            messages = self.ollamaChat.chatWithHistory(transcription, chatHistoryFile, self.system_prompt)
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

        def generateAudio(sentences, audioFile):
            while self.running == True:
                sentence = sentences.get()

                if sentence != done:
                    startTime = time.perf_counter()
                    
                    audioResponse = 'audio-response/response-' + os.urandom(8).hex() + '.wav' 
                    self.textToSpeech.textToSpeech(sentence, audioResponse, self.tts_cuda)
                    
                    endTime = time.perf_counter()
                    elapsedTime = endTime - startTime
                    print(f"Took {elapsedTime} seconds to generate audio")

                    audioFile.put(audioResponse)
                
                else:
                    audioFile.put(done)
                    print("Finished generating audio")
                    break

        def sendAudioResponseToClient(audioFile):
            while self.running == True:
                audioResponse = audioFile.get()
                
                if audioResponse != done:
                    file_size = os.path.getsize(audioResponse)
                    self.client_socket.sendall(str(file_size).encode())

                    with open(audioResponse, 'rb') as f:
                        self.client_socket.recv(1024) # Not sure why but adding this stops erroneous wave data being sent with file_size causing a crash.
                        received_size = 0
                        while self.running == True:
                            chunk = f.read(1024)

                            if not chunk:
                                break

                            self.client_socket.sendall(chunk)

                    self.client_socket.recv(1024) # Wait for client to finish writing to file

                    if os.path.exists(audioResponse):
                        os.remove(audioResponse)
                
                else:
                    print("Finished sending audio")
                    self.client_socket.sendall("end".encode())
                    break

        # Create a queue
        sentences = queue.Queue()
        audioFile = queue.Queue()

        # Create and start the threads
        sendTextResponseToClient = threading.Thread(target=sendTextResponseToClient, args=(sentences,))
        sendTextResponseToClient.start()
        generateAudio = threading.Thread(target=generateAudio, args=(sentences, audioFile,))
        generateAudio.start()
        sendAudioResponseToClient = threading.Thread(target=sendAudioResponseToClient, args=(audioFile,))
        sendAudioResponseToClient.start()

        # Wait for the threads to finish
        sendTextResponseToClient.join()
        generateAudio.join()
        sendAudioResponseToClient.join()

    def receiveAudioInput(self, audioInput):
        file_size = self.client_socket.recv(1024).decode()
        
        if file_size == "Voice was not detected":
            print("Voice was not detected")
            audioResponse = 'audio-response/response-' + os.urandom(8).hex() + '.wav' 
            self.textToSpeech.textToSpeech(text='I\'m sorry, I didn\'t catch that.', fileName=audioResponse, cuda=self.tts_cuda)
            self.sendResponseToClient(audioResponse)
            self.handleClient()

        if file_size == "endOfSpeech":
            return False

        else:
            self.client_socket.sendall('ACK'.encode())

            # Open a file to write the received data
            with open(audioInput, 'wb') as f:
                received_size = 0
                
                while received_size < int(file_size):
                    chunk = self.client_socket.recv(1024)

                    f.write(chunk)

                    received_size += len(chunk)

        return True

    def start(self):
        while True:
            self.client_socket, address = self.server_socket.accept()
            print(f'Connected by {address}')
            client_handler = threading.Thread(target=self.handleClient)
            client_handler.start()

if __name__ == '__main__':
        server = llmVoiceAssistantServer()
        server.start()
