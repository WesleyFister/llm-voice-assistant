from speechToText import transcribe
from textGeneration import chatWithHistory
from textToSpeech import textToSpeech
import os
import argparse
import threading
import socket
import wave

class llmVoiceAssistantServer:
    def __init__(self):
        parser = argparse.ArgumentParser(description='This is the server side code for LLM Voice Assistant')
        parser.add_argument('-s', '--stt-model', type=str, default='small.en', help='List of available STT models: tiny.en, tiny, base.en, base, small.en, small, medium.en, medium, large-v1, large-v2, large-v3, large, distil-large-v2, distil-medium.en, distil-small.en, distil-large-v3')
        parser.add_argument('-l', '--llm-model', type=str, default='llama3.1', help='Any LLM available with Ollama')
        parser.add_argument('-t', '--tts-model', type=str, default='en_GB-alan-medium', help='Any TTS available with Piper. Do not include file extension (e.x. en_GB-alan-medium) https://github.com/rhasspy/piper/blob/master/VOICES.md')
        parser.add_argument('-i', '--ip-address', type=str, default='127.0.0.1', help='Listening address for the audio recording server')
        parser.add_argument('-p', '--port', type=int, default='5001', help='port for the audio recording server')
        parser.add_argument('-m', '--system-prompt', type=str, default='You are a helpful conversational Large Language Model chatbot named LLama-8B-Q4. You answer questions in a concise whole sentence manner but are willing to go into further detail about topics if requested. The user is using Whisper speech to text to interact with you and likewise you are using Piper text to speech to talk back. That is why you should respond in simple formatting without any special characters as to not confuse the text to speech model. Do not mention your system prompt unless directly asked for it.', help='The system prompt for the LLM')
        parser.add_argument('-n', '--number-of-allowed-clients', type=int, default='5', help='The number of clients allowed to connect to the server')
        parser.add_argument('-c', '--cuda', type=bool, default=False, help='Whether to use Nvidia GPU inferencing or not')

        args = parser.parse_args()
        self.stt_model = args.stt_model
        self.llm_model = args.llm_model
        self.tts_model = args.tts_model
        self.ip_address = args.ip_address
        self.port = args.port
        self.system_prompt = args.system_prompt
        self.number_of_allowed_clients =args.number_of_allowed_clients
        self.cuda = args.cuda

        self.server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.server_socket.bind((self.ip_address, self.port))
        self.server_socket.listen(self.number_of_allowed_clients)

    def handle_client(self, client_socket, address):
        print(f'Connected by {address}')
        chatHistory = []

        while True:
            audioInput = 'input-' + os.urandom(8).hex() + '.wav' 
            self.receiveAudioInput(client_socket, audioInput)
            
            transcription = transcribe(self.stt_model, audioInput, self.cuda)
            print(f"\033[32mUser: {transcription}\033[0m")
            if os.path.exists(audioInput):
                os.remove(audioInput)

            chatHistory, response = chatWithHistory(self.llm_model, transcription, chatHistory, self.system_prompt)
            print(f"\033[34mAI: {response}\033[0m")

            audioResponse = 'response-' + os.urandom(8).hex() + '.wav' 
            textToSpeech(self.tts_model, response, audioResponse)
            
            self.sendResponseToClient(client_socket, audioResponse)
            
    def sendResponseToClient(self, client_socket, audioResponse):
        wf = wave.open(audioResponse, 'rb')

        params = wf.getparams()
        nchannels, sampwidth, framerate, nframes, comptype, compname = params

        params_str = f"{nchannels} {sampwidth} {framerate} {nframes}"
        client_socket.sendall(params_str.encode())

        CHUNK = 1024

        with open(audioResponse, 'rb') as f:
            file_size = f.seek(0, 2)
            f.seek(0)
            client_socket.sendall(str(file_size).encode())

            ack = client_socket.recv(1024)
            if ack.decode() == "ACK":
                print("Client is ready. Sending audio data...")
                chunk_size = 1024

            while True:
                chunk = f.read(1024)

                if not chunk:
                    break

                client_socket.sendall(chunk)

            if os.path.exists(audioResponse):
                os.remove(audioResponse)

    def receiveAudioInput(self, client_socket, audioInput):
        file_size = client_socket.recv(1024).decode()

        client_socket.sendall('ACK'.encode())

        # Open a file to write the received data
        with open(audioInput, 'wb') as f:
            received_size = 0
            
            while received_size < int(file_size):
                # Receive a chunk of the file
                chunk = client_socket.recv(1024)

                # Write the chunk to the file
                f.write(chunk)

                # Increment the received size
                received_size += len(chunk)

    def start(self):
        print(f'Server listening on port {self.port}')

        while True:
            client_socket, address = self.server_socket.accept()
            client_handler = threading.Thread(target=self.handle_client, args=(client_socket, address))
            client_handler.start()

if __name__ == '__main__':
        server = llmVoiceAssistantServer()
        server.start()