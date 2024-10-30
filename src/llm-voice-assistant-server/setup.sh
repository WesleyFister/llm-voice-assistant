if [[ -z $(command -v ollama) ]]; then
    echo "Installing Ollama"
    curl -fsSL https://ollama.com/install.sh | sh
fi

if [ -x "$(command -v apt-get)" ]; then
    sudo apt-get install nvidia-cudnn -y
    sudo apt-get install pyenv
elif [ -x "$(command -v pacman)" ]; then
    sudo pacman -Sy cudnn
    sudo pacman -Sy pyenv
fi

pyenv install 3.11 --skip-existing
pyenv global 3.11
pyenv exec python3.11 -m venv venv

source venv/bin/activate

# MeloTTS
pip install git+https://github.com/myshell-ai/MeloTTS.git
python -m unidic download

pip install -r requirements.txt

# Faster-Whisper
#pip install git+https://github.com/SYSTRAN/faster-whisper.git


