import os
from os import path

HOME_DIR = '/home/utilisateur/Documents/test/'
TEMP_DIR = '/home/utilisateur/Documents/test/temp'

# Create the temporary directory if it does not exist
os.makedirs(TEMP_DIR, exist_ok=True)

# Change to the main working directory
os.chdir(HOME_DIR)

# Clone the sylva3D repository if it does not already exist
if not path.exists(f'{HOME_DIR}/sylva3D'):
    os.system('git clone https://github.com/Amiche02/sylva3D.git')

# Change to the sylva3D repository directory and update it
os.chdir(f'{HOME_DIR}/sylva3D')
os.system('git pull')

# Download and unzip the ckpts files from Google Drive if they do not already exist
if not path.exists(f'{TEMP_DIR}/ckpts'):
    os.chdir(TEMP_DIR)
    os.system('wget --no-check-certificate "https://drive.google.com/uc?export=download&id=17_ghtmKegxLwSI8J-69s7zGCMsnPlyIl" -O ckpts.zip')
    os.system('unzip ckpts.zip -d temp_ckpts')
    os.system('mv temp_ckpts/* ckpts')
    os.system('rm -r temp_ckpts')
    os.system('rm ckpts.zip')

# Create a symbolic link to ckpts in the sylva3D repository if it does not already exist
if not path.exists(f'{HOME_DIR}/sylva3D/ckpts'):
    os.system(f'ln -s {TEMP_DIR}/ckpts {HOME_DIR}/sylva3D/')

# Download the SAM model if necessary
if not path.exists(f'{HOME_DIR}/sylva3D/sam_pt'):
    os.chdir(f'{HOME_DIR}/sylva3D')
    os.makedirs('sam_pt', exist_ok=True)
    os.chdir(f'{HOME_DIR}/sylva3D/sam_pt')
    os.system('wget https://huggingface.co/segments-arnaud/sam_vit_h/resolve/main/sam_vit_h_4b8939.pth')

# Install openssh with mamba
os.system('mamba install openssh -y')

# Change back to the sylva3D repository directory
os.chdir(f'{HOME_DIR}/sylva3D')

# Install Python dependencies with pip
os.system('pip install typing_extensions')
os.system('pip install triton')
os.system('pip install fire')
os.system('pip install opencv-python')
os.system('pip install rembg')
os.system('pip install git+https://github.com/facebookresearch/segment-anything.git')
os.system('pip install streamlit')
os.system('pip install -r requirements.txt')
os.system('pip install gdown')

# Download a file with gdown
os.system('gdown "https://drive.google.com/u/1/uc?id=1-7x7qQfB7bIw2zV4Lr6-yhvMpjXC84Q5&confirm=t"')

# Download and install tinycudann if necessary
if not path.exists(f'tinycudann-1.7-cp310-cp310-linux_x86_64.whl'):
    os.system('wget "https://j2q5.c17.e2-1.dev/download/pogscafe/tinycudann-1.7-cp310-cp310-linux_x86_64.whl"')
os.system('pip install tinycudann-1.7-cp310-cp310-linux_x86_64.whl')

# Install gradio and update torch and xformers
os.system('pip install gradio==3.48.0')
os.system('pip uninstall -y xformers')
os.system('pip install --no-cache-dir torch==2.1.0 torchvision==0.16.0 torchaudio==2.1.0 xformers --index-url https://download.pytorch.org/whl/cu118')
