read -p "Would you like to install Ollama? (y/N): " ollama
if [ $ollama = "y" ];
then
if [[ -z $(command -v ollama) ]]; then
    echo "Installing Ollama"
    curl -fsSL https://ollama.com/install.sh | sh
fi
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

pip install -r requirements.txt

# You have to uninstall onnxruntime first before installing onnxruntime-gpu.
pip uninstall onnxruntime -y
pip install "onnxruntime-gpu==1.20.1"
