from faster_whisper import WhisperModel

def transcribe(stt_model, audioInput, cuda):
	transcript = ""

	if cuda == True:
		# or run on GPU with INT8
		model = WhisperModel(stt_model, device="cuda", compute_type="int8_float16")

	else:
		# or run on CPU with INT8
		model = WhisperModel(stt_model, device="cpu", compute_type="int8")

	segments, info = model.transcribe(audioInput, beam_size=5, language="en")
	for segment in segments:
		transcript += segment.text
	
	return transcript