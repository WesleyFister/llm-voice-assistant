Carry a spoken conversation with LMM models! This project uses Whisper speech to text to transcribe the user's voice, send it to the LMM and pipe the final result to Piper text to speech. This project uses a socket server to get audio from a client device and sends it to the server to do all the processing. The final text to speech output from the LLM is sent to the client device.

### WIP Warning
This is very much a work in progress. Many basic features have yet to be implemented.

### Install
This program only works on Linux Ubuntu/Debian and Arch systems.

Run 'setup.sh' script for both the client and server side. Next run 'start.sh' for the server first followed by the client next.

### Features
- Fully offline and local
- Wake word detection: 'Hey Jarvis'
- 100% hands free interation
- Client server model
- Streamed reponses
- Multilingual speech-to-text and large-language-model

### Configuring
In both the client and server configuring is done by opening 'start.sh' and passing in the corresponding flag to 'main.py' like so.
`python3 main.py --ip-address 192.168.1.123 --port 5432`

### Todo
1. Cancel audio playback when saying the wake word 'hey Jarvis'.
2. Properly close the server with CTRL + C.
3. Make a proper setup.sh for both the client and server.
4. Clear LLM chat history by saying some variation of 'hey Jarvis clear chat history'.
5. Allow multiple clients to connect to the server.
6. Clean up code.
7. Add multilingual TTS (OpenVoice).
