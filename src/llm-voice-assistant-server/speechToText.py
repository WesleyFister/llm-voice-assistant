from faster_whisper import WhisperModel
from faster_whisper.utils import download_model

class speechToText:
    def __init__(self, stt_model, cuda):
        self.stt_model = stt_model
        self.cuda = cuda
        
        download_model(self.stt_model)

        if self.cuda == True:
            # or run on GPU with INT8
            print('Running Whisper on GPU inferencing')
            self.model = WhisperModel(self.stt_model, device="cuda", compute_type="int8")

        else:
            # or run on CPU with INT8
            print('Running Whisper on CPU inferencing')
            self.model = WhisperModel(self.stt_model, device="cpu", compute_type="int8")

    def transcribe(self, audioInput):
        transcript = ""

        segments, info = self.model.transcribe(audioInput, beam_size=5) # language="en"
        print("Detected language '%s' with probability %f" % (info.language, info.language_probability))
        for segment in segments:
            transcript += segment.text

        return { "transcript": transcript, "language": info.language }
