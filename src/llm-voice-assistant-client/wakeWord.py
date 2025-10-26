from openwakeword.model import Model
import openwakeword
import pyaudio
import numpy as np

def wakeWord(model):
    if model == "alexa":
        openwakeword.utils.download_models(['alexa_v0.1'])
        model = Model(wakeword_models = ["alexa"])

    elif model == "hey mycroft":
        openwakeword.utils.download_models(['hey_mycroft_v0.1'])
        model = Model(wakeword_models = ["hey mycroft"])

    elif model == "hey jarvis":
        openwakeword.utils.download_models(['hey_jarvis_v0.1'])
        model = Model(wakeword_models = ["hey jarvis"])

    else:
        model = Model(wakeword_models = model)

    p = pyaudio.PyAudio()

    FORMAT = pyaudio.paInt16
    CHANNELS = 1
    RATE = 16000
    CHUNK = 1280

    stream = p.open(format=FORMAT, channels=CHANNELS, rate=RATE, input=True, frames_per_buffer=CHUNK)

    
    print("Listening for wake word...")

    sentinel = False
    while True:
        audio = np.frombuffer(stream.read(CHUNK, False), dtype=np.int16)
        prediction = model.predict(x=audio, threshold={'hey jarvis': 0.9}, debounce_time=5.0)

        for mdl in model.prediction_buffer.keys():
            scores = list(model.prediction_buffer[mdl])
            curr_score = format(scores[-1], '.20f').replace("-", "")
            
            if float(curr_score) >= 0.9:
                print("Wake word detected!")
                
                sentinel = True
                break
                
        if sentinel == True:
            stream.stop_stream()
            stream.close()
            break