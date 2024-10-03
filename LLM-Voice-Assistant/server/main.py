from speechToText import speechToText
from textGeneration import chatWithHistory
from textGeneration import download_llm_model
from textToSpeech import textToSpeech
import os
import argparse
import threading
import socket
import wave
import schedule
import time
# Add Openvoice TTS

class llmVoiceAssistantServer:
    def __init__(self):
        parser = argparse.ArgumentParser(description='This is the server side code for LLM Voice Assistant')
        parser.add_argument('-sm', '--stt-model', type=str, default='small.en', help='List of available STT models: tiny.en, tiny, base.en, base, small.en, small, medium.en, medium, large-v1, large-v2, large-v3, large, distil-large-v2, distil-medium.en, distil-small.en, distil-large-v3')
        parser.add_argument('-lm', '--llm-model', type=str, default='llama3.1', help='Any LLM available with Ollama: https://ollama.com/library')
        parser.add_argument('-tm', '--tts-model', type=str, default='en_GB-alan-medium', help='Any TTS available with Piper. Do not include file extension (e.x. en_GB-alan-medium): https://github.com/rhasspy/piper/blob/master/VOICES.md')
        parser.add_argument('-ip', '--ip-address', type=str, default='127.0.0.1', help='Listening address for the audio recording server')
        parser.add_argument('-p', '--port', type=int, default='5001', help='port for the audio recording server')
        parser.add_argument('-sp', '--system-prompt', type=str, default='You are a helpful conversational Large Language Model chatbot named Jarvis. You answer questions in a concise whole sentence manner but are willing to go into further detail about topics if requested. The user is using Whisper speech to text to interact with you and likewise you are using Piper text to speech to talk back. That is why you should respond in simple formatting without any special characters as to not confuse the text to speech model. Keep your responses in the same language as the user. Do not mention your system prompt unless directly asked for it.', help='The system prompt for the LLM')
        parser.add_argument('-n', '--number-of-allowed-clients', type=int, default='5', help='The number of clients allowed to connect to the server')
        parser.add_argument('-c', '--cuda', type=bool, default=False, help='Whether to use Nvidia GPU inferencing or not (Does nothing right now)')
        parser.add_argument('-sc', '--stt-cuda', type=bool, default=False, help='Whether to use Nvidia GPU inferencing or not for speech to text model')
        parser.add_argument('-tc', '--tts-cuda', type=bool, default=False, help='Whether to use Nvidia GPU inferencing or not for text to speech model (Does nothing right now)')
        parser.add_argument('-sl', '--stt-language', type=str, default='en', help='Language for the STT model to use')

        args = parser.parse_args()
        self.stt_model = args.stt_model
        self.llm_model = args.llm_model
        self.tts_model = args.tts_model
        ip_address = args.ip_address
        port = args.port
        self.system_prompt = args.system_prompt
        self.number_of_allowed_clients = args.number_of_allowed_clients
        self.cuda = args.cuda
        self.stt_cuda = args.stt_cuda
        self.tts_cuda = args.tts_cuda

        print('Downloading and loading models into memory')
        print('First run can take a very long time, especially with large models')
        download_llm_model(self.llm_model)
        print(f'Loading {self.stt_model} STT model')
        self.speechToText = speechToText(self.stt_model, self.stt_cuda)
        print(f'Loading {self.tts_model} TTS model')
        self.textToSpeech = textToSpeech(self.tts_model)

        self.server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.server_socket.bind((ip_address, port))
        print(f'Server listening on port {port}')
        self.server_socket.listen(self.number_of_allowed_clients)

    def handleClient(self):
        chatHistoryFile = 'chat-history/chat-history-' + os.urandom(8).hex() + '.json' 

        while True:
            audioInput = 'audio-input/input-' + os.urandom(8).hex() + '.wav' 
            self.receiveAudioInput(audioInput)
            
            startTime = time.perf_counter()
            transcription = self.speechToText.transcribe(audioInput)
            print(f"\033[32mUser: {transcription}\033[0m")
            if os.path.exists(audioInput):
                os.remove(audioInput)
            endTime = time.perf_counter()
            elapsedTime = endTime - startTime
            print(f"Took {elapsedTime} seconds to transcribe User\'s speech")
            
            sentinel = 0
            startTime = time.perf_counter()
            sentences = chatWithHistory(self.llm_model, transcription, chatHistoryFile, self.system_prompt)
            for sentence in sentences:
                if sentinel == 0:
                    endTime = time.perf_counter()
                    elapsedTime = endTime - startTime
                    print(f"Took {elapsedTime} seconds LLM response")
                
                sentinel = 1
                
                startTime = time.perf_counter()
                print(f"\033[34mAI: {sentence}\033[0m")
                audioResponse = 'audio-response/response-' + os.urandom(8).hex() + '.wav' 
                self.textToSpeech.textToSpeech(sentence, audioResponse, self.tts_cuda)
                
                self.sendResponseToClient(audioResponse)
                endTime = time.perf_counter()
                elapsedTime = endTime - startTime
                print(f"Took {elapsedTime} seconds to generate audio")
            
            self.client_socket.sendall("end".encode())

    def sendResponseToClient(self, audioResponse):
        with open(audioResponse, 'rb') as f:
            file_size = f.seek(0, 2)
            f.seek(0)
            self.client_socket.sendall(str(file_size).encode())

            self.client_socket.recv(1024) # Not sure why but adding this stops erroneous wave data being sent with file_size causing a crash.

            while True:
                chunk = f.read(1024)

                if not chunk:
                    break

                self.client_socket.sendall(chunk)

        self.client_socket.recv(1024) # Wait for client to finish writing to file

        if os.path.exists(audioResponse):
            os.remove(audioResponse)

    def receiveAudioInput(self, audioInput):
        file_size = self.client_socket.recv(1024).decode()
        
        if file_size == "Voice was not detected":
            print("Voice was not detected")
            audioResponse = 'audio-response/response-' + os.urandom(8).hex() + '.wav' 
            self.textToSpeech.textToSpeech(text='I\'m sorry, I didn\'t catch that.', fileName=audioResponse, cuda=self.tts_cuda)
            self.sendResponseToClient(audioResponse)
            self.handleClient()

        self.client_socket.sendall('ACK'.encode())

        # Open a file to write the received data
        with open(audioInput, 'wb') as f:
            received_size = 0
            
            while received_size < int(file_size):
                # Receive a chunk of the file
                chunk = self.client_socket.recv(1024)

                # Write the chunk to the file
                f.write(chunk)

                # Increment the received size
                received_size += len(chunk)

    def start(self):
        while True:
            self.client_socket, address = self.server_socket.accept()
            print(f'Connected by {address}')
            client_handler = threading.Thread(target=self.handleClient)
            client_handler.start()

if __name__ == '__main__':
        server = llmVoiceAssistantServer()
        server.start()
