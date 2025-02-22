# Linux :

### create and activate the conda enviroment:
```bash
conda create -n sylv3D python=3.10 -y
conda activate sylv3D
```

### create and activate the venv enviroment:
```bash
sudo apt-get install python3-venv

python3 -m venv sylv3D
source sylv3D/bin/activate
```

If you don't want to use a new virtual environent, then uninstall the old libraries to get the right version: 
```bash
chmod +x uninstall_requirements.sh
./uninstall_requirements.sh
```

## use python to setup all the dependencies:

```python
import os
from os import path

HOME_DIR = '/home/utilisateur/Documents/test'
TEMP_DIR = path.join(HOME_DIR, 'temp')

# Create the temporary directory if it does not exist
os.makedirs(TEMP_DIR, exist_ok=True)

# Clone the sylva3D repository if it does not already exist
os.chdir(HOME_DIR)
if not path.exists(path.join(HOME_DIR, 'sylva3D')):
    os.system('git clone https://github.com/Amiche02/sylva3D.git')

# Change to the sylva3D repository directory and update it
os.chdir(path.join(HOME_DIR, 'sylva3D'))
os.system('git pull')

# Download and unzip the ckpts files from Hugging Face if they do not already exist
os.chdir(TEMP_DIR)
ckpts_zip_path = path.join(TEMP_DIR, 'ckpts.zip')
if not path.exists(path.join(HOME_DIR, 'sylva3D', 'ckpts')):
    os.system(f'wget https://huggingface.co/Amiche02/wonder3D/resolve/main/ckpts.zip -O {ckpts_zip_path}')
    os.system(f'unzip {ckpts_zip_path} -d {path.join(HOME_DIR, "sylva3D")}')
    os.remove(ckpts_zip_path)

# Create a symbolic link to ckpts in the sylva3D repository if it does not already exist
ckpts_symlink_path = path.join(HOME_DIR, 'sylva3D', 'ckpts')
if not path.exists(ckpts_symlink_path):
    os.system(f'ln -s {path.join(HOME_DIR, "sylva3D", "ckpts")} {ckpts_symlink_path}')

# Download the SAM model if necessary
sam_pt_dir = path.join(HOME_DIR, 'sylva3D', 'sam_pt')
if not path.exists(sam_pt_dir):
    os.makedirs(sam_pt_dir, exist_ok=True)
    os.system(f'wget https://huggingface.co/segments-arnaud/sam_vit_h/resolve/main/sam_vit_h_4b8939.pth -O {path.join(sam_pt_dir, "sam_vit_h_4b8939.pth")}')

# Install openssh with mamba
if os.system('command -v mamba') == 0:
    os.system('mamba install openssh -y')
else:
    print("mamba command not found. Please install mamba and try again.")

# Change back to the sylva3D repository directory
os.chdir(path.join(HOME_DIR, 'sylva3D'))

# Uninstall all the preinstalled libraries to make sure the installation is clean
uninstall_script = path.join(HOME_DIR, 'sylva3D', 'uninstall_requirements.sh')
if path.exists(uninstall_script):
    os.system('chmod +x uninstall_requirements.sh')
    os.system('./uninstall_requirements.sh')
else:
    print(f"{uninstall_script} does not exist")

# Install missing dependencies to avoid conflicts
os.system('pip install matplotlib torch torchvision')

# Install Python dependencies with pip
dependencies = [
    'typing_extensions', 'triton', 'fire', 'opencv-python', 'rembg',
    'git+https://github.com/facebookresearch/segment-anything.git', '-r requirements.txt', 'gdown'
]
for dep in dependencies:
    os.system(f'pip install {dep}')

# Download a file with gdown
os.system('gdown "https://drive.google.com/u/1/uc?id=1-7x7qQfB7bIw2zV4Lr6-yhvMpjXC84Q5&confirm=t"')

# Download and install tinycudann if necessary
tinycudann_whl = 'tinycudann-1.7-cp310-cp310-linux_x86_64.whl'
if not path.exists(path.join(HOME_DIR, 'sylva3D', tinycudann_whl)):
    os.system(f'wget "https://j2q5.c17.e2-1.dev/download/pogscafe/{tinycudann_whl}" -O {path.join(HOME_DIR, "sylva3D", tinycudann_whl)}')
os.system(f'pip install {path.join(HOME_DIR, "sylva3D", tinycudann_whl)}')

# Install gradio and update torch and xformers
os.system('pip install gradio==3.48.0')
os.system('pip uninstall -y xformers')
os.system('pip install --no-cache-dir torch==2.1.0 torchvision==0.16.0 torchaudio==2.1.0 xformers --index-url https://download.pytorch.org/whl/cu118')

```


## install manually the right version :

clone the repository
```bash
git clone https://github.com/Amiche02/sylva3D.git
cd sylva3D
```
Download and unzip the ckpts files from Hugging Face if they do not already exist : ---> use the right path to correctly unzip the file
```bash
wget https://huggingface.co/Amiche02/wonder3D/resolve/main/ckpts.zip
unzip /home/{path}/sylva3D/ckpts.zip

rm -r /home/{path}/sylva3D/ckpts.zip
```

Download the SAM model for segmentation
```bash
mkdir sam_pt
cd sam_pt
wget https://huggingface.co/segments-arnaud/sam_vit_h/resolve/main/sam_vit_h_4b8939.pth
cd ..
```
install all the required packages
```bash
pip install -r requirements.txt
```

## The usage of the script would look like this:
#### with graphical interface:

#### 3D reconstruction :
```sh
python3 app_recon.py
```

#### without graphical interface:
```sh
python3 inference.py --input_image /root/sylva3D/example_images/cat_head.png --output_dir /root/sylva3D/outputs --guidance_scale 2.0 --steps 100 --seed 42 --write_image
```

#### 3D reconstruction :
```sh
python3 inference_recon.py --input_image /root/sylva3D/example_images/cat_head.png --output_dir /root/sylva3D/outputs --guidance_scale 2.0 --steps 100 --seed 42 --write_image
```