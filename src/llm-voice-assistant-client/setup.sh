if [[ -z $(command -v ffmpeg) ]]; then
    sudo apt install ffmpeg -y
fi

if [ -x "$(command -v apt-get)" ]; then
    sudo apt-get install python3-venv -y
    sudo apt-get install python3-pip -y
    sudo apt-get install portaudio19-dev -y
    sudo apt-get install libpython3-dev -y
    sudo apt-get install pyenv -y
elif [ -x "$(command -v pacman)" ]; then
    sudo pacman -Sy pyenv
fi

pyenv install 3.11 --skip-existing
pyenv global 3.11.10
pyenv exec python3.11 -m venv venv

source venv/bin/activate
pip install -r requirements.txt

# If not using Pulseaudio
# sudo apt install pulseaudio -y
# reboot
