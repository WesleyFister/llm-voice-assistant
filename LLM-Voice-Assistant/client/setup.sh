if [[ -z $(command -v ffmpeg) ]]; then
    echo "Installing FFmpeg"
    sudo apt install ffmpeg -y
fi

sudo apt install python3-venv -y
sudo apt install python3-pip -y
sudo apt install portaudio19-dev -y
sudo apt-get install libpython3-dev -y

python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# If not using Pulseaudio
# sudo apt install pulseaudio -y
# reboot
