source venv/bin/activate
export LD_LIBRARY_PATH=`python3 -c 'import os; import nvidia.cublas.lib; import nvidia.cudnn.lib; print(os.path.dirname(nvidia.cublas.lib.__file__) + ":" + os.path.dirname(nvidia.cudnn.lib.__file__))'`
python3 main.py --stt-cuda --tts-cuda --stt-model large-v3-turbo --llm-api 192.168.1.100:9292 --llm-model Mistral-Small-3.2-24B-Instruct-2506:Q4_K_M
