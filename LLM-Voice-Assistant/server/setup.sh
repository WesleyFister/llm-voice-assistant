if [[ -z $(command -v ollama) ]]; then
    echo "Installing Ollama"
    curl -fsSL https://ollama.com/install.sh | sh
fi

sudo apt install nvidia-cudnn -y
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
