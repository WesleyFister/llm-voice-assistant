from wakeWord import wakeWord
import os
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

class llmVoiceAssistantClient():
    def __init__(self):
        parser = argparse.ArgumentParser(description='This is the client side code for LLM Voice Assistant')
        parser.add_argument('-i', '--ip-address', type=str, default='127.0.0.1', help='Listening address for the audio recording server')
        parser.add_argument('-p', '--port', type=int, default='5001', help='port for the audio recording server')
        parser.add_argument('-v', '--vad-initial-delay', type=float, default='10', help='The delay in seconds before audio stops recording when no voice is detected ()')
        parser.add_argument('-d', '--vad-delay', type=float, default='2', help='The delay in seconds before audio stops recording when a person stops speaking')

        args = parser.parse_args()
        ip_address = args.ip_address
        port = args.port
        
        # num_samples / SAMPLE_RATE = seconds of audio
        # In this case it is 0.032 seconds.
        mult = 31.25
        self.vad_initial_delay = int(args.vad_initial_delay * mult)
        self.vad_delay = int(args.vad_delay * mult)

        os.makedirs("audio-input", exist_ok=True)
        os.makedirs("audio-response", exist_ok=True)

        self.client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        print(f'Listening for {ip_address} on port {port}')

        self.client_socket.connect((ip_address, port))
        print('Connected to server')

    def clientStart(self):
        play_obj = None

        while True:
            wakeWord()
            self.running = False
            self.client_socket.sendall("0".encode())
            if play_obj != None:
                play_obj.stop()

            audioInput = self.recordAudio()
            self.sendAudioInput(audioInput)
            self.running = True
            getAudio_thread = threading.Thread(target=self.playAudioResponse)
            getAudio_thread.start()

    def recordAudio(self):
        # Download and load the Silero VAD model.
        model, utils = torch.hub.load(repo_or_dir='snakers4/silero-vad', model='silero_vad')

        # Provided by Alexander Veysov.
        def int2float(sound):
            abs_max = np.abs(sound).max()
            sound = sound.astype('float32')
            if abs_max > 0:
                sound *= 1/32768
                sound = sound.squeeze()  # depends on the use case.
                return sound

        audio = pyaudio.PyAudio()
        
        FORMAT = pyaudio.paInt16
        CHANNELS = 1
        SAMPLE_RATE = 16000
        CHUNK = int(SAMPLE_RATE / 10)

        num_samples = 512
        stream = audio.open(format=FORMAT, channels=CHANNELS, rate=SAMPLE_RATE, input=True, frames_per_buffer=CHUNK)

        print("Started Recording")
        # Load and play an audio file.
        audioFile = sa.WaveObject.from_wave_file("media/on.wav")
        play_obj = audioFile.play()

        audioInput = 'audio-input/input-' + os.urandom(8).hex() + '.wav' 
        wf = wave.open(audioInput, 'wb')
        wf.setnchannels(CHANNELS)
        wf.setsampwidth(audio.get_sample_size(FORMAT))
        wf.setframerate(SAMPLE_RATE)

        silence = 0
        voiceDetected = False
        delay = self.vad_initial_delay
        new_confidence = 1

        # Send user audio data to the server.
        while silence < delay:
            print(f"\x1b[2K{silence}/{delay}", end="\r") # \x1b[2K ANSI sequence for clearing the current line.
            silence += 1
            audio_chunk = stream.read(num_samples)

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
                silence = 0
                voiceDetected = True
                
        wf.close()
        print(f'{silence}/{delay}')
        print("Stopped recording")
        # Load and play an audio file.
        audioFile = sa.WaveObject.from_wave_file("media/off.wav")
        play_obj = audioFile.play()

        if voiceDetected == False:
            print("Voice was not detected")
            self.client_socket.sendall("Voice was not detected".encode())
            self.playAudioResponse()
            self.clientStart()

        return audioInput
        
    def sendAudioInput(self, audioInput):
        with open(audioInput, 'rb') as f:
            file_size = f.seek(0, 2)
            f.seek(0)

            self.client_socket.sendall(str(file_size).encode())

            ack = self.client_socket.recv(1024)
            if ack.decode()!= 'ACK':
                print('Error: Client did not acknowledge file size')
                self.client_socket.close()
                exit()

            # Send the file to the server.
            while True:
                chunk = f.read(1024)

                # If the end of the file is reached, break the loop.
                if not chunk:
                    break

                self.client_socket.sendall(chunk)

            if os.path.exists(audioInput):
                os.remove(audioInput)

    def playAudioResponse(self):
        def getAudio(q):
            while True:
                file_size = self.client_socket.recv(1024)
                if file_size.decode() == "end":
                    print("End of output")
                    q.put("end")
                    break

                self.client_socket.sendall("ACK".encode()) # Not sure why but adding this stops erroneous wave data being sent with file_size causing a crash.

                audioResponse = 'audio-response/response-' + os.urandom(8).hex() + '.wav' 

                # Open a file to write the received data
                with open(audioResponse, 'wb') as f:
                    received_size = 0
                    
                    while received_size < int(file_size):
                        # Receive a chunk of the file
                        chunk = self.client_socket.recv(1024)

                        # Write the chunk to the file
                        f.write(chunk)

                        # Increment the received size
                        received_size += len(chunk)

                q.put(audioResponse)

                self.client_socket.sendall("ACK".encode())

        def playAudio(q):
            while True:
                audioResponse = q.get()
                if audioResponse == 'end':
                    break

                if self.running == True:
                    audioFile = sa.WaveObject.from_wave_file(audioResponse)
                    play_obj = audioFile.play()
                    play_obj.wait_done() # Makes program wait for audio to finish.

                if os.path.exists(audioResponse):
                    os.remove(audioResponse)

        # Create a queue
        q = queue.Queue()

        # Create and start the producer thread
        getAudio_thread = threading.Thread(target=getAudio, args=(q,))
        getAudio_thread.start()

        # Create and start the consumer thread
        playAudio_thread = threading.Thread(target=playAudio, args=(q,))
        playAudio_thread.start()

        # Wait for the producer to finish
        getAudio_thread.join()

        # Wait for the consumer to finish
        playAudio_thread.join()

if __name__ == '__main__':
    client = llmVoiceAssistantClient()
    client.clientStart()
