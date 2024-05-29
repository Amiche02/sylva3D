Linux :

## use python to setup all the dependencies:

create and activate the conda enviroment:
```bash
conda create -n newenv python=3.10 -y
conda activate newenv
```

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
if not path.exists(ckpts_zip_path):
    os.system(f'wget https://huggingface.co/Amiche02/wonder3D/resolve/main/ckpts.zip -O {ckpts_zip_path}')
    os.system(f'unzip {ckpts_zip_path} -d {HOME_DIR}')
    os.remove(ckpts_zip_path)

# Create a symbolic link to ckpts in the sylva3D repository if it does not already exist
ckpts_symlink_path = path.join(HOME_DIR, 'sylva3D', 'ckpts')
if not path.exists(ckpts_symlink_path):
    os.system(f'ln -s {path.join(HOME_DIR, "ckpts")} {ckpts_symlink_path}')

# Download the SAM model if necessary
sam_pt_dir = path.join(HOME_DIR, 'sylva3D', 'sam_pt')
if not path.exists(sam_pt_dir):
    os.makedirs(sam_pt_dir, exist_ok=True)
    os.system(f'wget https://huggingface.co/segments-arnaud/sam_vit_h/resolve/main/sam_vit_h_4b8939.pth -O {path.join(sam_pt_dir, "sam_vit_h_4b8939.pth")}')

# Install openssh with mamba
os.system('mamba install openssh -y')

# Change back to the sylva3D repository directory
os.chdir(path.join(HOME_DIR, 'sylva3D'))

# Install Python dependencies with pip
dependencies = [
    'typing_extensions', 'triton', 'fire', 'opencv-python', 'rembg',
    'git+https://github.com/facebookresearch/segment-anything.git', 'streamlit', '-r requirements.txt', 'gdown'
]
for dep in dependencies:
    os.system(f'pip install {dep}')

# Download a file with gdown
os.system('gdown "https://drive.google.com/u/1/uc?id=1-7x7qQfB7bIw2zV4Lr6-yhvMpjXC84Q5&confirm=t"')

# Download and install tinycudann if necessary
tinycudann_whl = 'tinycudann-1.7-cp310-cp310-linux_x86_64.whl'
if not path.exists(tinycudann_whl):
    os.system(f'wget "https://j2q5.c17.e2-1.dev/download/pogscafe/{tinycudann_whl}"')
os.system(f'pip install {tinycudann_whl}')

# Install gradio and update torch and xformers
os.system('pip install gradio==3.48.0')
os.system('pip uninstall -y xformers')
os.system('pip install --no-cache-dir torch==2.1.0 torchvision==0.16.0 torchaudio==2.1.0 xformers --index-url https://download.pytorch.org/whl/cu118')
```


Or use directly command line :
```bash
sudo apt-get install gcc-9 g++-9
sudo update-alternatives --install /usr/bin/gcc gcc /usr/bin/gcc-9 60 --slave /usr/bin/g++ g++ /usr/bin/g++-9

export CC=/usr/bin/gcc-9
export CXX=/usr/bin/g++-9

sudo apt-get update
sudo apt-get install build-essential ninja-build


pip install git+https://github.com/NVlabs/tiny-cuda-nn/#subdirectory=bindings/torch
```

colab :
```bash
curl -L "https://github.com/OutofAi/tiny-cuda-nn-wheels/releases/download/1.7.0/tinycudann-1.7+arch75+torch221+cuda121-cp310-cp310-linux_x86_64.whl" -o tinycudann-1.7+arch75+torch221+cuda121-cp310-cp310-linux_x86_64.whl
pip install tinycudann-1.7+arch75+torch221+cuda121-cp310-cp310-linux_x86_64.whl --force-reinstall
import tinycudann as tcnn
```

uninstall old libraries : 
```bash
chmod +x uninstall_requirements.sh
./uninstall_requirements.sh
```

install the right version :
```bash
pip install -r requirements.txt
pip install fire streamlit
```

The usage of the script would look like this:
```sh
python inference.py --input_image /root/sylva3D/example_images/cat_head.png --output_dir /root/sylva3D/outputs --guidance_scale 2.0 --steps 100 --seed 42 --write_image
```

3D reconstruction :
```sh
python inference_recon.py --input_image /root/sylva3D/example_images/cat_head.png --output_dir /root/sylva3D/outputs --guidance_scale 2.0 --steps 100 --seed 42 --write_image
```