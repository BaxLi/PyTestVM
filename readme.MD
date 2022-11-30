sudo apt-get update -y
sudo apt-get install -y python-digitalocean
python3 -V

#let’s install pip,
sudo apt install -y python3-pip

#Python packages can be installed by typing:
pip3 install package_name

#Installing virtualenv
python3 -m pip install --user virtualenv

#Installing packages
python3 -m pip install **package-name**
python3 -m pip install -r requirements.txt

#Creating a virtual environment
python3 -m venv env
source env/bin/activate
