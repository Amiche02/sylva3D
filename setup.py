import os
from os import path

HOME_DIR = '/home/utilisateur/Téléchargements/test/working'
TEMP_DIR = '/home/utilisateur/Téléchargements/test/temp'

os.makedirs(TEMP_DIR, exist_ok=True)

os.chdir(HOME_DIR)

if not path.exists(f'{HOME_DIR}/sylva3D'):
    os.system('git clone https://github.com/Amiche02/sylva3D.git')

os.chdir(f'{HOME_DIR}/sylva3D')
os.system('git pull')

if not path.exists(f'{TEMP_DIR}/ckpts'):
    os.chdir(TEMP_DIR)
    os.system('git clone https://huggingface.co/camenduru/Wonder3D')
    os.system('mv Wonder3D ckpts')

if not path.exists(f'{HOME_DIR}/sylva3D/ckpts'):
    os.system(f'ln -s {TEMP_DIR}/ckpts {HOME_DIR}/sylva3D/')

if not path.exists(f'{HOME_DIR}/sylva3D/sam_pt'):
    os.chdir(f'{HOME_DIR}/sylva3D')
    os.makedirs('sam_pt', exist_ok=True)
    os.chdir(f'{HOME_DIR}/sylva3D/sam_pt')
    os.system('wget https://huggingface.co/segments-arnaud/sam_vit_h/resolve/main/sam_vit_h_4b8939.pth')

os.system('mamba install openssh -y')

os.chdir(f'{HOME_DIR}/sylva3D')
os.system('conda create -n newenv python=3.10 -y')
os.system('rm -rf /opt/conda/lib && cp -r /opt/conda/envs/newenv/lib /opt/conda')

os.system('pip install typing_extensions')
os.system('pip install triton')
os.system('pip install fire')
os.system('pip install opencv-python')
os.system('pip install rembg')
os.system('pip install git+https://github.com/facebookresearch/segment-anything.git')
os.system('pip install streamlit')
os.system('pip install -r requirements.txt')
os.system('pip install gdown')

os.system('gdown "https://drive.google.com/u/1/uc?id=1-7x7qQfB7bIw2zV4Lr6-yhvMpjXC84Q5&confirm=t"')

if not path.exists(f'tinycudann-1.7-cp310-cp310-linux_x86_64.whl'):
    os.system('wget "https://j2q5.c17.e2-1.dev/download/pogscafe/tinycudann-1.7-cp310-cp310-linux_x86_64.whl"')

os.system('pip install tinycudann-1.7-cp310-cp310-linux_x86_64.whl')
os.system('pip install gradio==3.48.0')
os.system('pip uninstall -y xformers')
os.system('pip install --no-cache-dir torch==2.1.0 torchvision==0.16.0 torchaudio==2.1.0 xformers --index-url https://download.pytorch.org/whl/cu118')
