# 1. Update and install prerequisites
sudo apt update
sudo apt install -y software-properties-common

# 2. Add the deadsnakes PPA (which contains newer Python releases)
sudo add-apt-repository ppa:deadsnakes/ppa
sudo apt update

# 3. Install Python 3.12 along with development and venv packages
sudo apt install -y python3.12 python3.12-dev python3.12-venv

# 4. (Optional) Install pip for Python 3.12
curl -sS https://bootstrap.pypa.io/get-pip.py -o get-pip.py
python3.12 get-pip.py
rm get-pip.py

# 5. Configure update-alternatives so that python3 points to Python 3.12
# (Assuming your current default is Python 3.8; adjust if necessary.)
sudo update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.8 1
sudo update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.12 2

# 6. Choose the default python3 (select Python 3.12 from the menu)
sudo update-alternatives --config python3

# 7. Verify the change
python3 --version
